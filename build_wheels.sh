#!/bin/bash
set -e -x


# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install cython
    "${PYBIN}/pip" install -r /io/requirements-full.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install aliyun-odps-python-sdk --no-index -f /io/wheelhouse
done
