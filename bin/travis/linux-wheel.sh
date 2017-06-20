#!/bin/bash
set -e -x
PYBIN=/opt/python/${PYVER}/bin
${PYBIN}/pip install --disable-pip-version-check --user --upgrade pip
${PYBIN}/pip install cython
cd /io/
# Compile wheels
${PYBIN}/python setup.py bdist_wheel

for whl in dist/*.whl; do
	auditwheel repair $whl -w dist/
done

rm dist/*-linux*.whl
