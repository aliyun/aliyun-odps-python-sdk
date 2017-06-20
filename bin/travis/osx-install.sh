set -e

brew update
# Per the `pyenv homebrew recommendations <https://github.com/yyuu/pyenv/wiki#suggested-build-environment>`_.
brew install openssl readline
# See https://docs.travis-ci.com/user/osx-ci-environment/#A-note-on-upgrading-packages.
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
PYENV_ROOT="$HOME/.pyenv"
PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"


if [${PYTHON} -le "3.0"]; then
	curl -O https://bootstrap.pypa.io/get-pip.py
	python get-pip.py --user
else
	pyenv install $PYTHON
	pyenv global $PYTHON
fi

#check python version
python -V
pip install cython
pip install wheel

pip wheel --no-deps .
#repair_wheel
mkdir dist
cp *.whl dist/
pip install delocate
delocate-wheel dist/*.whl
delocate-addplat --rm-orig -x 10_9 -x 10_10 dist/*.whl

