ODPS Python SDK
===============

Elegent way to access ODPS API.
`Documentation <http://pyodps.readthedocs.org/>`__

Installation
------------

The quick way:

::

   pip install pyodps[full]

If you don’t need to use Jupyter, just type

::

   pip install pyodps

The dependencies will be installed automatically.

Or from source code (not recommended for production use):

.. code:: shell

   $ virtualenv pyodps_env
   $ source pyodps_env/bin/activate
   $ pip install git+https://github.com/aliyun/aliyun-odps-python-sdk.git

Dependencies
------------

- Python (>=2.7), including Python 3+, pypy, Python 3.7 recommended
- setuptools (>=3.0)

Run Tests
---------

- install pytest
- copy conf/test.conf.template to odps/tests/test.conf, and fill it with
  your account
- run ``pytest odps``

Usage
-----

.. code:: python

   >>> import os
   >>> from odps import ODPS
   >>> # Make sure environment variable CLOUD_ACCESS_KEY_ID already set to Access Key ID of user
   >>> # while environment variable CLOUD_ACCESS_KEY_SECRET set to Access Key Secret of user.
   >>> # Not recommended to hardcode Access Key ID or Access Key Secret in your code.
   >>> o = ODPS(
   >>>     os.getenv('CLOUD_ACCESS_KEY_ID'),
   >>>     os.getenv('CLOUD_ACCESS_KEY_SECRET'),
   >>>     project='**your-project**',
   >>>     endpoint='**your-endpoint**',
   >>> )
   >>> dual = o.get_table('dual')
   >>> dual.name
   'dual'
   >>> dual.table_schema
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
   >>> dual.table_schema.columns
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

Command-line and IPython enhancement
------------------------------------

::

   In [1]: %load_ext odps

   In [2]: %enter
   Out[2]: <odps.inter.Room at 0x10fe0e450>

   In [3]: %sql select * from pyodps_iris limit 5
   |==========================================|   1 /  1  (100.00%)         2s
   Out[3]:
      sepallength  sepalwidth  petallength  petalwidth         name
   0          5.1         3.5          1.4         0.2  Iris-setosa
   1          4.9         3.0          1.4         0.2  Iris-setosa
   2          4.7         3.2          1.3         0.2  Iris-setosa
   3          4.6         3.1          1.5         0.2  Iris-setosa
   4          5.0         3.6          1.4         0.2  Iris-setosa

Python UDF Debugging Tool
-------------------------

.. code:: python

   #file: plus.py
   from odps.udf import annotate

   @annotate('bigint,bigint->bigint')
   class Plus(object):
       def evaluate(self, a, b):
           return a + b

::

   $ cat plus.input
   1,1
   3,2
   $ pyou plus.Plus < plus.input
   2
   5

Contributing
------------

For a development install, clone the repository and then install from
source:

::

   git clone https://github.com/aliyun/aliyun-odps-python-sdk.git
   cd pyodps
   pip install -r requirements.txt -e .

License
-------

Licensed under the `Apache License
2.0 <https://www.apache.org/licenses/LICENSE-2.0.html>`__
