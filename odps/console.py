#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

import sys
import multiprocessing
import threading
import collections
import time
import struct
import os
import math
import locale
import codecs
import warnings

try:
    import fcntl
    import termios
    import signal
    _CAN_RESIZE_TERMINAL = True
except ImportError:
    _CAN_RESIZE_TERMINAL = False

try:
    from IPython import get_ipython
except ImportError:
    pass
try:
    get_ipython()
except NameError:
    OutStream = None
    IPythonIOStream = None
    widgets = None
else:
    from IPython import version_info
    ipython_major_version = version_info[0]

    try:
        from ipykernel.iostream import OutStream
    except ImportError:
        try:
            from IPython.zmq.iostream import OutStream
        except ImportError:
            if ipython_major_version < 4:
                try:
                    from IPython.kernel.zmq.iostream import OutStream
                except ImportError:
                    OutStream = None
            else:
                OutStream = None

    if OutStream is not None:
        from IPython.utils import io as ipyio
        # On Windows in particular this is necessary, as the io.stdout stream
        # in IPython gets hooked up to some pyreadline magic to handle colors
        IPythonIOStream = ipyio.IOStream
    else:
        OutStream = None
        IPythonIOStream = None

    # On Windows, in IPython 2 the standard I/O streams will wrap
    # pyreadline.Console objects if pyreadline is available; this should
    # be considered a TTY
    try:
        from pyreadyline.console import Console as PyreadlineConsole
    except ImportError:
        # Just define a dummy class
        class PyreadlineConsole(object): pass

    # import widgets and display
    try:
        if ipython_major_version < 4:
            from IPython.html import widgets
        else:
            from ipywidgets import widgets
    except ImportError:
        widgets = None
    from IPython.display import display

    # ignore widgets deprecated message
    def _ignore_deprecated_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'.*widget.*')

    if get_ipython and get_ipython():
        get_ipython().events.register('pre_execute', _ignore_deprecated_warnings)

from .compat import six
from .config import options


_DEFAULT_ENCODING = 'utf-8'
_initial_defencoding = None


def detect_console_encoding():
    """
    Try to find the most capable encoding supported by the console.
    slighly modified from the way IPython handles the same issue.
    """
    import locale
    global _initial_defencoding

    encoding = None
    try:
        encoding = sys.stdout.encoding or sys.stdin.encoding
    except AttributeError:
        pass

    # try again for something better
    if not encoding or 'ascii' in encoding.lower():
        try:
            encoding = locale.getpreferredencoding()
        except Exception:
            pass

    # when all else fails. this will usually be "ascii"
    if not encoding or 'ascii' in encoding.lower():
        encoding = sys.getdefaultencoding()

    # GH3360, save the reported defencoding at import time
    # MPL backends may change it. Make available for debugging.
    if not _initial_defencoding:
        _initial_defencoding = sys.getdefaultencoding()

    return encoding


def get_terminal_size():
    """
    Detect terminal size and return tuple = (width, height).

    Only to be used when running in a terminal. Note that the IPython notebook,
    IPython zmq frontends, or IDLE do not run in a terminal,
    """
    import platform
    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os == 'Linux' or \
        current_os == 'Darwin' or \
            current_os.startswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)      # default value
    return tuple_xy


def in_interactive_session():
    """ check if we're running in an interactive shell

    returns True if running under python/ipython interactive shell
    """
    def check_main():
        import __main__ as main
        return not hasattr(main, '__file__')

    try:
        return __IPYTHON__ or check_main()
    except:
        return check_main()


def in_ipython_frontend():
    """
    check if we're inside an an IPython zmq frontend
    """
    try:
        ip = get_ipython()
        return 'zmq' in str(type(ip)).lower()
    except:
        pass

    return False


def is_widgets_available():
    if widgets is None:
        return False
    if hasattr(widgets.Widget, '_version_validated'):
        return bool(getattr(widgets.Widget, '_version_validated', None))
    else:
        return True


def in_qtconsole():
    """
    check if we're inside an IPython qtconsole

    DEPRECATED: This is no longer needed, or working, in IPython 3 and above.
    """
    try:
        ip = get_ipython()
        front_end = (
            ip.config.get('KernelApp', {}).get('parent_appname', "") or
            ip.config.get('IPKernelApp', {}).get('parent_appname', "")
        )
        if 'qtconsole' in front_end.lower():
            return True
    except:
        return False
    return False


def get_console_size():
    """Return console size as tuple = (width, height).

    Returns (None,None) in non-interactive session.
    """
    display_width = options.display.width
    # deprecated.
    display_height = options.display.max_rows

    # Consider
    # interactive shell terminal, can detect term size
    # interactive non-shell terminal (ipnb/ipqtconsole), cannot detect term
    # size non-interactive script, should disregard term size

    # in addition
    # width,height have default values, but setting to 'None' signals
    # should use Auto-Detection, But only in interactive shell-terminal.
    # Simple. yeah.

    if in_interactive_session():
        if in_ipython_frontend():
            # sane defaults for interactive non-shell terminal
            # match default for width,height in config_init
            try:
                from pandas.core.config import get_default_val
                terminal_width = get_default_val('display.width')
                terminal_height = get_default_val('display.max_rows')
            except ImportError:
                terminal_width, terminal_height = None, None
        else:
            # pure terminal
            terminal_width, terminal_height = get_terminal_size()
    else:
        terminal_width, terminal_height = None, None

    # Note if the User sets width/Height to None (auto-detection)
    # and we're in a script (non-inter), this will return (None,None)
    # caller needs to deal.
    return (display_width or terminal_width, display_height or terminal_height)


def _get_terminal_size_windows():
    res = None
    try:
        from ctypes import windll, create_string_buffer

        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12

        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
    except:
        return None
    if res:
        import struct
        (bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx,
         maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
        sizex = right - left + 1
        sizey = bottom - top + 1
        return sizex, sizey
    else:
        return None


def _get_terminal_size_tput():
    # get terminal width
    # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width
    # -height-of-a-terminal-window
    try:
        import subprocess
        proc = subprocess.Popen(["tput", "cols"],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        output = proc.communicate(input=None)
        cols = int(output[0])
        proc = subprocess.Popen(["tput", "lines"],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        output = proc.communicate(input=None)
        rows = int(output[0])
        return (cols, rows)
    except:
        return None


def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios
            import struct
            import os
            cr = struct.unpack(
                'hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except:
            return None
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr or cr == (0, 0):
        try:
            from os import environ as env
            cr = (env['LINES'], env['COLUMNS'])
        except:
            return None
    return int(cr[1]), int(cr[0])


def _get_stdout(stderr=False):
    """
    This utility function contains the logic to determine what streams to use
    by default for standard out/err.

    Typically this will just return `sys.stdout`, but it contains additional
    logic for use in IPython on Windows to determine the correct stream to use
    (usually ``IPython.util.io.stdout`` but only if sys.stdout is a TTY).
    """

    if stderr:
        stream = 'stderr'
    else:
        stream = 'stdout'

    sys_stream = getattr(sys, stream)

    if IPythonIOStream is None:
        return sys_stream

    ipyio_stream = getattr(ipyio, stream)

    if isatty(sys_stream) and isatty(ipyio_stream):
        # Use the IPython console output stream
        return ipyio_stream
    else:
        # sys.stdout was set to some other non-TTY stream (a file perhaps)
        # so just use it directly
        return sys_stream


def isatty(file):
    """
    Returns `True` if `file` is a tty.

    Most built-in Python file-like objects have an `isatty` member,
    but some user-defined types may not, so this assumes those are not
    ttys.
    """
    if (multiprocessing.current_process().name != 'MainProcess' or
        threading.current_thread().getName() != 'MainThread'):
        return False

    if hasattr(file, 'isatty'):
        return file.isatty()
    elif (OutStream is not None and
          isinstance(file, (OutStream, IPythonIOStream)) and
          ((hasattr(file, 'name') and file.name == 'stdout') or
           (hasattr(file, 'stream') and
               isinstance(file.stream, PyreadlineConsole)))):
        # File is an IPython OutStream or IOStream and
        #    File name is 'stdout' or
        #    File wraps a Console
        return True
    return False


def _terminal_size(file=None):
    """
    Returns a tuple (height, width) containing the height and width of
    the terminal.

    This function will look for the width in height in multiple areas
    before falling back on the width and height in astropy's
    configuration.
    """

    if file is None:
        file = _get_stdout()

    try:
        s = struct.pack(str("HHHH"), 0, 0, 0, 0)
        x = fcntl.ioctl(file, termios.TIOCGWINSZ, s)
        (lines, width, xpixels, ypixels) = struct.unpack(str("HHHH"), x)
        if lines > 12:
            lines -= 6
        if width > 10:
            width -= 1
        return (lines, width)
    except:
        try:
            # see if POSIX standard variables will work
            return (int(os.environ.get('LINES')),
                    int(os.environ.get('COLUMNS')))
        except TypeError:
            # fall back on configuration variables, or if not
            # set, (25, 80)
            lines = options.console.max_lines
            width = options.console.max_width
            if lines is None:
                lines = 25
            if width is None:
                width = 80
            return lines, width


def _color_text(text, color):
    """
    Returns a string wrapped in ANSI color codes for coloring the
    text in a terminal::

        colored_text = color_text('Here is a message', 'blue')

    This won't actually effect the text until it is printed to the
    terminal.

    Parameters
    ----------
    text : str
        The string to return, bounded by the color codes.
    color : str
        An ANSI terminal color name. Must be one of:
        black, red, green, brown, blue, magenta, cyan, lightgrey,
        default, darkgrey, lightred, lightgreen, yellow, lightblue,
        lightmagenta, lightcyan, white, or '' (the empty string).
    """
    color_mapping = {
        'black': '0;30',
        'red': '0;31',
        'green': '0;32',
        'brown': '0;33',
        'blue': '0;34',
        'magenta': '0;35',
        'cyan': '0;36',
        'lightgrey': '0;37',
        'default': '0;39',
        'darkgrey': '1;30',
        'lightred': '1;31',
        'lightgreen': '1;32',
        'yellow': '1;33',
        'lightblue': '1;34',
        'lightmagenta': '1;35',
        'lightcyan': '1;36',
        'white': '1;37'}

    if sys.platform == 'win32' and OutStream is None:
        # On Windows do not colorize text unless in IPython
        return text

    color_code = color_mapping.get(color, '0;39')
    return '\033[{0}m{1}\033[0m'.format(color_code, text)


def _decode_preferred_encoding(s):
    """Decode the supplied byte string using the preferred encoding
    for the locale (`locale.getpreferredencoding`) or, if the default encoding
    is invalid, fall back first on utf-8, then on latin-1 if the message cannot
    be decoded with utf-8.
    """

    enc = locale.getpreferredencoding()
    try:
        try:
            return s.decode(enc)
        except LookupError:
            enc = _DEFAULT_ENCODING
        return s.decode(enc)
    except UnicodeDecodeError:
        return s.decode('latin-1')


def _write_with_fallback(s, write, fileobj):
    """Write the supplied string with the given write function like
    ``write(s)``, but use a writer for the locale's preferred encoding in case
    of a UnicodeEncodeError.  Failing that attempt to write with 'utf-8' or
    'latin-1'.
    """

    if IPythonIOStream is not None and isinstance(fileobj, IPythonIOStream):
        # If the output stream is an IPython.utils.io.IOStream object that's
        # not going to be very helpful to us since it doesn't raise any
        # exceptions when an error occurs writing to its underlying stream.
        # There's no advantage to us using IOStream.write directly though;
        # instead just write directly to its underlying stream:
        write = fileobj.stream.write

    try:
        write(s)
        return write
    except UnicodeEncodeError:
        # Let's try the next approach...
        pass

    enc = locale.getpreferredencoding()
    try:
        Writer = codecs.getwriter(enc)
    except LookupError:
        Writer = codecs.getwriter(_DEFAULT_ENCODING)

    f = Writer(fileobj)
    write = f.write

    try:
        write(s)
        return write
    except UnicodeEncodeError:
        Writer = codecs.getwriter('latin-1')
        f = Writer(fileobj)
        write = f.write

    # If this doesn't work let the exception bubble up; I'm out of ideas
    write(s)
    return write


def color_print(*args, **kwargs):
    """
    Prints colors and styles to the terminal uses ANSI escape
    sequences.

    ::

       color_print('This is the color ', 'default', 'GREEN', 'green')

    Parameters
    ----------
    positional args : str
        The positional arguments come in pairs (*msg*, *color*), where
        *msg* is the string to display and *color* is the color to
        display it in.

        *color* is an ANSI terminal color name.  Must be one of:
        black, red, green, brown, blue, magenta, cyan, lightgrey,
        default, darkgrey, lightred, lightgreen, yellow, lightblue,
        lightmagenta, lightcyan, white, or '' (the empty string).

    file : writeable file-like object, optional
        Where to write to.  Defaults to `sys.stdout`.  If file is not
        a tty (as determined by calling its `isatty` member, if one
        exists), no coloring will be included.

    end : str, optional
        The ending of the message.  Defaults to ``\\n``.  The end will
        be printed after resetting any color or font state.
    """

    file = kwargs.get('file', _get_stdout())

    end = kwargs.get('end', '\n')

    write = file.write
    if isatty(file) and options.console.use_color:
        for i in range(0, len(args), 2):
            msg = args[i]
            if i + 1 == len(args):
                color = ''
            else:
                color = args[i + 1]

            if color:
                msg = _color_text(msg, color)

            # Some file objects support writing unicode sensibly on some Python
            # versions; if this fails try creating a writer using the locale's
            # preferred encoding. If that fails too give up.
            if not six.PY3 and isinstance(msg, bytes):
                msg = _decode_preferred_encoding(msg)

            write = _write_with_fallback(msg, write, file)

        write(end)
    else:
        for i in range(0, len(args), 2):
            msg = args[i]
            if not six.PY3 and isinstance(msg, bytes):
                # Support decoding bytes to unicode on Python 2; use the
                # preferred encoding for the locale (which is *sometimes*
                # sensible)
                msg = _decode_preferred_encoding(msg)
            write(msg)
        write(end)


def human_time(seconds):
    """
    Returns a human-friendly time string that is always exactly 6
    characters long.

    Depending on the number of seconds given, can be one of::

        1w 3d
        2d 4h
        1h 5m
        1m 4s
          15s

    Will be in color if console coloring is turned on.

    Parameters
    ----------
    seconds : int
        The number of seconds to represent

    Returns
    -------
    time : str
        A human-friendly representation of the given number of seconds
        that is always exactly 6 characters.
    """
    units = [
        ('y', 60 * 60 * 24 * 7 * 52),
        ('w', 60 * 60 * 24 * 7),
        ('d', 60 * 60 * 24),
        ('h', 60 * 60),
        ('m', 60),
        ('s', 1),
    ]

    seconds = int(seconds)

    if seconds < 60:
        return '   {0:2d}s'.format(seconds)
    for i in range(len(units) - 1):
        unit1, limit1 = units[i]
        unit2, limit2 = units[i + 1]
        if seconds >= limit1:
            return '{0:2d}{1}{2:2d}{3}'.format(
                seconds // limit1, unit1,
                (seconds % limit1) // limit2, unit2)
    return '  ~inf'


def human_file_size(size):
    """
    Returns a human-friendly string representing a file size
    that is 2-4 characters long.

    For example, depending on the number of bytes given, can be one
    of::

        256b
        64k
        1.1G

    Parameters
    ----------
    size : int
        The size of the file (in bytes)

    Returns
    -------
    size : str
        A human-friendly representation of the size of the file
    """
    suffixes = ' kMGTPEZY'
    if size == 0:
        num_scale = 0
    else:
        num_scale = int(math.floor(math.log(size) / math.log(1000)))
    num_scale = max(num_scale, 0)
    if num_scale > 7:
        suffix = '?'
    else:
        suffix = suffixes[num_scale]
    num_scale = int(math.pow(1000, num_scale))
    value = float(size) / num_scale
    str_value = str(value)
    if suffix == ' ':
        str_value = str_value[:str_value.index('.')]
    elif str_value[2] == '.':
        str_value = str_value[:2]
    else:
        str_value = str_value[:3]
    return "{0:>3s}{1}".format(str_value, suffix)


class ProgressBar(six.Iterator):
    """
    A class to display a progress bar in the terminal.

    It is designed to be used either with the ``with`` statement::

        with ProgressBar(len(items)) as bar:
            for item in enumerate(items):
                bar.update()

    or as a generator::

        for item in ProgressBar(items):
            item.process()
    """
    def __init__(self, total_or_items, ipython_widget=False, file=None):
        """
        Parameters
        ----------
        total_or_items : int or sequence
            If an int, the number of increments in the process being
            tracked.  If a sequence, the items to iterate over.

        ipython_widget : bool, optional
            If `True`, the progress bar will display as an IPython
            notebook widget.

        file : writable file-like object, optional
            The file to write the progress bar to.  Defaults to
            `sys.stdout`.  If `file` is not a tty (as determined by
            calling its `isatty` member, if any, or special case hacks
            to detect the IPython console), the progress bar will be
            completely silent.
        """
        if ipython_widget:
            # Import only if ipython_widget, i.e., widget in IPython
            # notebook
            try:
                if ipython_major_version < 4:
                    from IPython.html import widgets
                else:
                    from ipywidgets import widgets
                from IPython.display import display
                ipython_widget = is_widgets_available()
            except ImportError:
                ipython_widget = False

        if file is None:
            file = _get_stdout()

        if not isatty(file) and not ipython_widget:
            self.update = self._silent_update
            self._silent = True
        else:
            self._silent = False

        if isinstance(total_or_items, collections.Iterable):
            self._items = iter(total_or_items)
            self._total = len(total_or_items)
        else:
            try:
                self._total = int(total_or_items)
            except TypeError:
                raise TypeError("First argument must be int or sequence")
            else:
                self._items = iter(range(self._total))

        self._file = file
        self._start_time = time.time()
        self._human_total = human_file_size(self._total)
        self._ipython_widget = ipython_widget


        self._signal_set = False
        if not ipython_widget:
            self._should_handle_resize = (
                _CAN_RESIZE_TERMINAL and self._file.isatty())
            self._handle_resize()
            if self._should_handle_resize:
                signal.signal(signal.SIGWINCH, self._handle_resize)
                self._signal_set = True

        self.update(0)

    def _handle_resize(self, signum=None, frame=None):
        terminal_width = _terminal_size(self._file)[1]
        self._bar_length = terminal_width - 37

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._silent:
            if exc_type is None:
                self.update(self._total)
            self._file.write('\n')
            self._file.flush()
            if self._signal_set:
                signal.signal(signal.SIGWINCH, signal.SIG_DFL)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            rv = next(self._items)
        except StopIteration:
            self.__exit__(None, None, None)
            raise
        else:
            self.update()
            return rv

    def update(self, value=None):
        """
        Update progress bar via the console or notebook accordingly.
        """

        # Update self.value
        if value is None:
            value = self._current_value + 1
        self._current_value = value

        # Choose the appropriate environment
        if self._ipython_widget:
            self._update_ipython_widget(value)
        else:
            self._update_console(value)

    def close(self):
        if not self._ipython_widget:
            self.__exit__(None, None, None)
        if self._ipython_widget and self._widget:
            self._widget.close()

    def _update_console(self, value=None):
        """
        Update the progress bar to the given value (out of the total
        given to the constructor).
        """

        if self._total == 0:
            frac = 1.0
        else:
            frac = float(value) / float(self._total)

        file = self._file
        write = file.write

        if frac > 1:
            bar_fill = int(self._bar_length)
        else:
            bar_fill = int(float(self._bar_length) * frac)
        write('\r|')
        color_print('=' * bar_fill, 'blue', file=file, end='')
        if bar_fill < self._bar_length:
            color_print('>', 'green', file=file, end='')
            write('-' * (self._bar_length - bar_fill - 1))
        write('|')

        if value >= self._total:
            t = time.time() - self._start_time
            prefix = '     '
        elif value <= 0:
            t = None
            prefix = ''
        else:
            t = ((time.time() - self._start_time) * (1.0 - frac)) / frac
            prefix = ' ETA '
        write(' {0:>4s}/{1:>4s}'.format(
            human_file_size(value),
            self._human_total))
        write(' ({0:>6s}%)'.format('{0:.2f}'.format(frac * 100.0)))
        write(prefix)
        if t is not None:
            write(human_time(t))
        self._file.flush()

    def _update_ipython_widget(self, value=None):
        """
        Update the progress bar to the given value (out of a total
        given to the constructor).

        This method is for use in the IPython notebook 2+.
        """

        # Create and display an empty progress bar widget,
        # if none exists.
        if not hasattr(self, '_widget'):
            # Import only if an IPython widget, i.e., widget in iPython NB
            if ipython_major_version < 4:
                self._widget = widgets.FloatProgressWidget()
            else:
                self._widget = widgets.FloatProgress()

            if is_widgets_available():
                display(self._widget)
            self._widget.value = 0

        # Calculate percent completion, and update progress bar
        percent = (float(value)/self._total) * 100.0
        self._widget.value = percent
        self._widget.description =' ({0:>6s}%)'.format('{0:.2f}'.format(percent))

    def _silent_update(self, value=None):
        pass

    @classmethod
    def map(cls, function, items, multiprocess=False, file=None):
        """
        Does a `map` operation while displaying a progress bar with
        percentage complete.

        ::

            def work(i):
                print(i)

            ProgressBar.map(work, range(50))

        Parameters
        ----------
        function : function
            Function to call for each step

        items : sequence
            Sequence where each element is a tuple of arguments to pass to
            *function*.

        multiprocess : bool, optional
            If `True`, use the `multiprocessing` module to distribute each
            task to a different processor core.

        file : writeable file-like object, optional
            The file to write the progress bar to.  Defaults to
            `sys.stdout`.  If `file` is not a tty (as determined by
            calling its `isatty` member, if any), the scrollbar will
            be completely silent.
        """

        results = []

        if file is None:
            file = _get_stdout()

        with cls(len(items), file=file) as bar:
            step_size = max(200, bar._bar_length)
            steps = max(int(float(len(items)) / step_size), 1)
            if not multiprocess:
                for i, item in enumerate(items):
                    results.append(function(item))
                    if (i % steps) == 0:
                        bar.update(i)
            else:
                p = multiprocessing.Pool()
                for i, result in enumerate(
                    p.imap_unordered(function, items, steps)):
                    bar.update(i)
                    results.append(result)
                p.close()
                p.join()

        return results


class StatusLine(object):
    def __init__(self, ipython_widget=False):
        if widgets is None:
            ipython_widget = False

        self.ipython_widget = ipython_widget
        self._widget = None

    def update(self, text):
        if self.ipython_widget:
            if not self._widget:
                self._widget = widgets.HTML()
                if is_widgets_available():
                    display(self._widget)
            self._widget.value = text

    def close(self):
        if self._widget:
            self._widget.close()
