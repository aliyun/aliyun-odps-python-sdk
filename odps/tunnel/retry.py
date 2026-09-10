#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tunnel-specific retry strategy.

Tunnel retry semantics:

* 308 (SLOT_REASSIGNMENT) / 429 (FLOW_EXCEEDED): infinite exponential
  backoff capped at 64s per wait.
* 5xx (server errors, incl. 502/504): exponential backoff 1s..64s,
  at most 7 retries.
* other 4xx: no retry, raise immediately.
* non-tunnel exceptions (e.g. network IO errors): fall back to the
  caller-supplied default policy.
"""

import logging
import threading
import time
import uuid

from .. import options
from ..errors import BaseODPSError

logger = logging.getLogger(__name__)

# HTTP status codes used by the tunnel protocol.
SLOT_REASSIGNMENT = 308
FLOW_EXCEEDED = 429
BAD_GATEWAY = 502
GATEWAY_TIMEOUT = 504
REQUEST_TIMEOUT = 408


# Header names used to correlate the requests in one retry sequence.
# The names deliberately avoid the ``x-odps-*`` prefix so they never
# enter the request signature (the canonical string only signs
# ``content-type`` / ``content-md5`` / ``x-odps-*`` headers).

#: Stable identifier shared by all requests in one retry sequence.
RETRY_TRACE_ID_HEADER = "odps-tunnel-retry-trace-id"

#: Zero-based request index within one retry sequence.
RETRY_INDEX_HEADER = "odps-tunnel-retry-index"


class RetryContext:
    """Identifies one actual request in an SDK retry sequence.

    The trace ID remains stable for the sequence, while ``retry_index`` is
    zero-based and increases for every subsequent request.  Calls to
    :meth:`next` share one thread-safe monotonic sequence and reserve
    distinct subsequent indexes.

    A context is one request snapshot and must not be reused as the first
    request of multiple retry loops.

    Parameters
    ----------
    trace_id : str, optional
        Stable sequence identifier; defaults to a fresh UUID.
    retry_index : int, optional
        Zero-based request index of the first request.
    """

    __slots__ = ("_trace_id", "_retry_index", "_sequence")

    _MISSING = object()

    def __init__(self, trace_id=_MISSING, retry_index=0, sequence=None):
        if trace_id is self._MISSING:
            trace_id = str(uuid.uuid4())
        if trace_id is None or not trace_id.strip():
            raise ValueError("Retry trace ID must not be blank")
        if retry_index < 0:
            raise ValueError("Retry index must not be negative")
        self._trace_id = trace_id
        self._retry_index = retry_index
        if sequence is None:
            sequence = {"value": retry_index, "lock": threading.Lock()}
        self._sequence = sequence

    @property
    def trace_id(self):
        """Stable trace ID shared across the retry sequence."""
        return self._trace_id

    @property
    def retry_index(self):
        """Zero-based index of this actual request."""
        return self._retry_index

    def next(self):
        """Return a context for the next actual request in the same sequence."""
        with self._sequence["lock"]:
            self._sequence["value"] += 1
            index = self._sequence["value"]
        return RetryContext(self._trace_id, index, self._sequence)

    def inject_headers(self, headers):
        """Add this request identity to the supplied mutable header dict.

        Parameters
        ----------
        headers : dict
            Mutable header map; gets ``RETRY_TRACE_ID_HEADER`` and
            ``RETRY_INDEX_HEADER`` entries added.
        """
        if headers is None:
            raise ValueError("headers must not be None")
        headers[RETRY_TRACE_ID_HEADER] = self._trace_id
        headers[RETRY_INDEX_HEADER] = str(self._retry_index)


# Backoff ceiling shared by both exponential policies (seconds).
# 2^6 = 64: the natural cap of the 2^(attempt-1) sequence. Beyond ~60s
# a retry backoff stops being useful for transient congestion; failing
# fast lets the caller surface the error rather than hanging a minute.
_MAX_BACKOFF = 64
# Max retry attempts for 5xx server errors.
# 7 doublings (attempts 1..7) reach the _MAX_BACKOFF ceiling; the total
# worst-case backoff is 1+2+4+8+16+32+64 = 127s before giving up.
_SERVER_ERROR_MAX_RETRIES = 7


class RetryPolicy:
    """Base retry policy. Subclasses override :meth:`should_retry`."""

    def should_retry(self, exc, attempt):
        raise NotImplementedError

    def get_retry_wait_time(self, attempt):
        raise NotImplementedError

    def wait_for_next_retry(self, attempt):
        time.sleep(self.get_retry_wait_time(attempt))


class NoRetryPolicy(RetryPolicy):
    """Never retry."""

    def should_retry(self, exc, attempt):
        return False

    def get_retry_wait_time(self, attempt):
        return 0

    def wait_for_next_retry(self, attempt):
        # No wait needed.
        pass


class InfiniteExponentialWaitRetryPolicy(RetryPolicy):
    """Infinite retries with exponential backoff capped at 64s.

    Used for 308 (slot reassignment) and 429 (flow exceeded).
    Wait sequence: 1, 2, 4, 8, 16, 32, 64, 64, 64, ...
    """

    def should_retry(self, exc, attempt):
        return True

    def get_retry_wait_time(self, attempt):
        if attempt < 7:
            return 2 ** (attempt - 1)
        return _MAX_BACKOFF


class ExponentialWaitRetryPolicy(RetryPolicy):
    """Bounded exponential backoff: at most 7 retries.

    Used for 5xx server errors. Wait sequence: 1, 2, 4, 8, 16, 32, 64.
    """

    def should_retry(self, exc, attempt):
        # attempt is 1-based; retry while attempt <= 7.
        return attempt <= _SERVER_ERROR_MAX_RETRIES

    def get_retry_wait_time(self, attempt):
        return 2 ** (attempt - 1)


class OptionsRetryPolicy(RetryPolicy):
    """Default retry policy for non-status exceptions (network IO).

    Mirrors ``utils.call_with_retry`` defaults so connection errors keep
    the historical retry behaviour when routed through
    ``TunnelRetryHandler`` instead of the generic helper.
    """

    def should_retry(self, exc, attempt):
        return attempt <= options.retry_times

    def get_retry_wait_time(self, attempt):
        return options.retry_delay


class TunnelRetryHandler:
    """Retry orchestrator that applies tunnel-specific policies.

    Parameters
    ----------
    default_retry_policy : RetryPolicy, optional
        Policy used for non-tunnel exceptions (e.g. network IO errors).
        Defaults to :class:`NoRetryPolicy`.
    retry_logger : callable, optional
        ``retry_logger(exc, attempt, wait_ms)`` invoked before each retry
        sleep.
    """

    def __init__(self, default_retry_policy=None, retry_logger=None):
        self._default_retry_policy = default_retry_policy or NoRetryPolicy()
        self._retry_logger = retry_logger

    def get_retry_policy(self, status_code):
        """Return the policy for a tunnel exception status code."""
        if status_code is None:
            return self._default_retry_policy
        if status_code in (SLOT_REASSIGNMENT, FLOW_EXCEEDED):
            return InfiniteExponentialWaitRetryPolicy()
        if status_code == REQUEST_TIMEOUT or 500 <= status_code < 600:
            return ExponentialWaitRetryPolicy()
        # Other 4xx: do not retry, regardless of default policy.
        return NoRetryPolicy()

    def _after_exception(self, exc, attempt, on_exception):
        """Decide whether to retry after *exc* and sleep if so.

        Returns ``True`` if the caller should retry, ``False`` if the
        exception should propagate (in which case it is re-raised here).
        """
        if isinstance(exc, BaseODPSError) and exc.status_code is not None:
            policy = self.get_retry_policy(exc.status_code)
        else:
            policy = self._default_retry_policy

        should_retry = policy.should_retry(exc, attempt)
        if not should_retry:
            if callable(on_exception) and not on_exception(exc):
                should_retry = True
            else:
                raise

        wait_ms = int(policy.get_retry_wait_time(attempt) * 1000)
        if self._retry_logger is not None:
            try:
                self._retry_logger(exc, attempt, wait_ms)
            except Exception:
                logger.debug("retry_logger callback raised", exc_info=True)
        try:
            policy.wait_for_next_retry(attempt)
        except KeyboardInterrupt:
            raise
        return True

    def execute_with_retry(self, action, on_error_status=None, on_exception=None):
        """Execute *action* with tunnel retry semantics.

        Parameters
        ----------
        action : callable
            Zero-argument callable performing the request.
        on_error_status : callable, optional
            ``on_error_status(status_code)`` invoked after each
            ``BaseODPSError`` is caught, before the retry decision.
        on_exception : callable, optional
            ``on_exception(exc)`` invoked when the retry policy decides
            not to retry.  If it returns ``True`` the exception is
            raised; if it returns ``False`` the exception is ignored
            and retrying continues.
        """
        return self.execute_with_retry_ctx(
            lambda ctx: action(),
            on_error_status=on_error_status,
            on_exception=on_exception,
        )

    def execute_with_retry_headers(
        self, request_func, headers, on_error_status=None, on_exception=None
    ):
        """Execute a request with tunnel retry, stamping correlation headers.

        Each attempt gets a fresh copy of *headers* with the
        ``odps-tunnel-retry-trace-id`` / ``odps-tunnel-retry-index``
        entries injected from the attempt's :class:`RetryContext`, so the
        server can correlate the requests in one SDK retry sequence. The
        caller-supplied *headers* map is never mutated.

        Parameters
        ----------
        request_func : callable
            ``request_func(stamped_headers) -> value``; performs the
            HTTP call with the supplied (already-stamped) header map.
        headers : dict
            Caller-supplied header map; copied and stamped per attempt.

        Returns
        -------
        object
            The return value of *request_func*.
        """
        base_headers = dict(headers) if headers is not None else {}

        def action(ctx):
            stamped = dict(base_headers)
            ctx.inject_headers(stamped)
            return request_func(stamped)

        return self.execute_with_retry_ctx(
            action,
            on_error_status=on_error_status,
            on_exception=on_exception,
        )

    def execute_with_retry_ctx(
        self,
        action,
        on_error_status=None,
        on_exception=None,
    ):
        """Execute a context-aware *action* with tunnel retry semantics.

        The zero-argument variant :meth:`execute_with_retry` behaves as if
        each actual request were wrapped in a freshly-created
        :class:`RetryContext`.  This variant instead hands the context to
        *action* so the request layer can stamp the
        ``odps-tunnel-retry-trace-id`` / ``odps-tunnel-retry-index``
        headers onto every attempt without mutating caller-supplied header
        maps.

        Parameters
        ----------
        action : callable
            ``action(retry_context) -> value``; receives the
            :class:`RetryContext` of the current actual request.
        on_error_status : callable, optional
            Same as :meth:`execute_with_retry`.
        on_exception : callable, optional
            Same as :meth:`execute_with_retry`.

        Returns
        -------
        object
            The return value of *action*.

        Raises
        ------
        Exception
            The last exception if no retry policy allows continuation.
        """
        context = RetryContext()
        attempt = 1
        while True:
            try:
                return action(context)
            except BaseODPSError as exc:
                if on_error_status is not None and exc.status_code is not None:
                    on_error_status(exc.status_code)
                self._after_exception(exc, attempt, on_exception)
                context = context.next()
                attempt += 1
            except Exception as exc:
                self._after_exception(exc, attempt, on_exception)
                context = context.next()
                attempt += 1
