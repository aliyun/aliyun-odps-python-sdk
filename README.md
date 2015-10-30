# ODPS Python SDK

Elegent way to access ODPS API.

## Dependencies

 * Python (=2.7.x)
 * requests (>= 2.1.0)
 * setuptools (>= 3.0)
 * protobuf (>=2.5.0)

## Installation

The quick way:

```
pip install git+http://github.com/aliyun/aliyun-odps-python-sdk.git
```

Or from source code:

```shell
$ virtualenv pyodps_env
$ source pyodps_env/bin/activate
$ git clone ...
$ cd pyodps
$ python setup.py install
```

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
>>> dual.create_time
datetime.datetime(2014, 6, 6, 13, 28, 24)
>>> dual.is_view
False
>>> dual.size
448
>>> dual.columns
[{u'comment': u'', u'type': u'string', u'name': u'id', u'label': u''}]
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
