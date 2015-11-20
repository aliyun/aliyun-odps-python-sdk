# ODPS Python SDK

[![PyPI version](https://badge.fury.io/py/pyodps.svg)](https://badge.fury.io/py/pyodps)

Elegent way to access ODPS API. [Documentation](http://pyodps.readthedocs.org)

## Installation

The quick way:

```
pip install pyodps
```
The dependencies will be installed automatically.

Or from source code:

```shell
$ virtualenv pyodps_env
$ source pyodps_env/bin/activate
$ git clone ...
$ cd pyodps
$ python setup.py install
```

## Dependencies

 * Python (>=2.6), including Python 3+, pypy
 * setuptools (>=3.0)
 * requests (>=2.4.0)
 * enum34 (>=1.0.4)
 * six (>=1.10.0)
 * protobuf (>=2.5.0)

## Run Unittest

- copy conf/test.conf.template to odps/test/test.conf, and fill it with your account
- run `python -m unittest discover`

## Usage

```python
>>> from odps import ODPS
>>> o = ODPS('**your-access-id**', '**your-secret-access-key**',
...          project='**your-project**', endpoint='**your-end-point**')
>>> dual = o.get_table('dual')
>>> dual.name
'dual'
>>> dual.schema
odps.Schema {
  c_int_a                 bigint
  c_int_b                 bigint
  c_double_a              double
  c_double_b              double
  c_string_a              string
  c_string_b              string
  c_bool_a                boolean
  c_bool_b                boolean
  c_datetime_a            datetime
  c_datetime_b            datetime
}
>>> dual.creation_time
datetime.datetime(2014, 6, 6, 13, 28, 24)
>>> dual.is_virtual_view
False
>>> dual.size
448
>>> dual.schema.columns
[<column c_int_a, type bigint>,
 <column c_int_b, type bigint>,
 <column c_double_a, type double>,
 <column c_double_b, type double>,
 <column c_string_a, type string>,
 <column c_string_b, type string>,
 <column c_bool_a, type boolean>,
 <column c_bool_b, type boolean>,
 <column c_datetime_a, type datetime>,
 <column c_datetime_b, type datetime>]
```

## Python UDF Debugging Tool

```python
#file: plus.py
from odps.udf import annotate

@annotate('bigint,bigint->bigint')
class Plus(object):
    def evaluate(self, a, b):
        return a + b
```

```
$ cat plus.input
1,1
3,2
$ pyou plus.Plus < plus.input
2
5
```

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)
