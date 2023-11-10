import logging

try:
    from sqlalchemy import Column
except ImportError:
    Column = None

try:
    from superset import sql_parse
    from superset.db_engine_specs.base import BaseEngineSpec
    from superset.exceptions import SupersetException
    from superset.extensions import cache_manager
except ImportError:
    # import fallback for tests only
    sql_parse = None

    class BaseEngineSpec(object):
        allows_sql_comments = True
        arraysize = 0

        @classmethod
        def get_engine(cls, database, schema=None, source=None):
            return database.get_sqla_engine_with_context(
                schema=schema, source=source
            )

        @classmethod
        def get_table_names(  # pylint: disable=unused-argument
            cls,
            database,
            inspector,
            schema,
        ):
            return set(inspector.get_table_names(schema))

        @classmethod
        def get_dbapi_mapped_exception(cls, ex):
            return ex

    class CacheManagerCls(object):
        def __init__(self):
            self.cache = self

        def memoize(self):
            return lambda x: x

    cache_manager = CacheManagerCls()

    class SupersetException(Exception):
        pass

try:
    from superset.constants import TimeGrain
except ImportError:
    # compatibility for older superset versions
    class TimeGrain:
        SECOND = "PT1S"
        MINUTE = "PT1M"
        HOUR = "PT1H"
        DAY = "P1D"
        WEEK = "P1W"
        MONTH = "P1M"
        QUARTER = "P3M"
        YEAR = "P1Y"
        WEEK_ENDING_SATURDAY = "P1W/1970-01-03T00:00:00Z"
        WEEK_STARTING_SUNDAY = "1969-12-28T00:00:00Z/P1W"


from .compat import six
from .config import options
from .df import DataFrame
from .utils import TEMP_TABLE_PREFIX

logger = logging.getLogger(__name__)

_builtin_funcs = set("""
ABS ACOS ADD_MONTHS ALL_MATCH ANY_MATCH ANY_VALUE ATAN2 APPROX_DISTINCT
ARG_MAX ARG_MIN ARRAY ARRAY_CONTAINS ARRAY_DISTINCT ARRAY_EXCEPT
ARRAY_INTERSECT ARRAY_JOIN ARRAY_MAX ARRAY_MIN ARRAY_NORMALIZE
ARRAY_POSITION ARRAY_REDUCE ARRAY_REMOVE ARRAY_REPEAT ARRAY_SORT
ARRAY_UNION ARRAYS_OVERLAP ARRAYS_ZIP ASCII ASIN ATAN AVG BASE64
BIN BITWISE_AND_AGG BITWISE_OR_AGG CAST CBRT CEIL CHAR_MATCHCOUNT
CHR CLUSTER_SAMPLE COALESCE COLLECT_LIST COLLECT_SET COMBINATIONS
COMPRESS CONCAT CONCAT_WS CONV CORR COS COSH COT COUNT COUNT_IF
COVAR_POP COVAR_SAMP CRC32 CUME_DIST CURRENT_TIMESTAMP CURRENT_TIMEZONE
DATE_ADD DATE_FORMAT DATE_SUB DATEADD DATEDIFF DATEPART DATETRUNC DAY
DAYOFMONTH DAYOFWEEK DAYOFYEAR DECODE DECOMPRESS DEGREES DENSE_RANK
E ENCODE EXP EXPLODE EXTRACT FACTORIAL FIELD FILTER FIRST_VALUE
FIND_IN_SET FLATTEN FLOOR FORMAT_NUMBER FROM_JSON FROM_UNIXTIME
FROM_UTC_TIMESTAMP GET_IDCARD_AGE GET_IDCARD_BIRTHDAY GET_IDCARD_SEX
GET_JSON_OBJECT GET_USER_ID GETDATE GREATEST HASH HEX HISTOGRAM
HOUR IF INDEX INLINE INITCAP INSTR IS_ENCODING ISDATE ISNAN JSON_OBJECT
JSON_ARRAY JSON_EXTRACT JSON_EXISTS JSON_PRETTY JSON_TYPE JSON_FORMAT
JSON_PARSE JSON_VALID JSON_TUPLE KEYVALUE KEYVALUE_TUPLE LAG
LAST_DAY LASTDAY LAST_VALUE LEAD LEAST LENGTH LENGTHB LN LOCATE
LOG LOG10 LOG2 LPAD LTRIM MAP MAP_AGG MAP_CONCAT MAP_ENTRIES
MAP_FILTER MAP_FROM_ARRAYS MAP_FROM_ENTRIES MAP_KEYS MAP_UNION
MAP_UNION_SUM MAP_VALUES MAP_ZIP_WITH MASK_HASH MAX MAX_BY MAX_PT
MD5 MEDIAN MIN MIN_BY MINUTE MONTH MONTHS_BETWEEN MULTIMAP_AGG
MULTIMAP_FROM_ENTRIES NAMED_STRUCT NEGATIVE NEXT_DAY NGRAMS NOW
NTILE NTH_VALUE NULLIF NUMERIC_HISTOGRAM NVL ORDINAL PARSE_URL
PARSE_URL_TUPLE PARTITION_EXISTS PERCENT_RANK PERCENTILE
PERCENTILE_APPROX PI POSEXPLODE POSITIVE POW QUARTER RADIANS
RAND RANK REGEXP_COUNT REGEXP_EXTRACT REGEXP_EXTRACT_ALL REGEXP_INSTR
REGEXP_REPLACE REGEXP_SUBSTR REPEAT REPLACE REVERSE ROUND ROW_NUMBER
RPAD RTRIM SAMPLE SECOND SEQUENCE SHA SHA1 SHA2 SHIFTLEFT SHIFTRIGHT
SHIFTRIGHTUNSIGNED SHUFFLE SIGN SIN SINH SIZE SLICE SORT_ARRAY
SOUNDEX SPACE SPLIT SPLIT_PART SQRT STACK STDDEV STDDEV_SAMP
STR_TO_MAP STRUCT SUBSTR SUBSTRING SUBSTRING_INDEX SUM
SYM_DECRYPT SYM_ENCRYPT TABLE_EXISTS TAN TANH TO_CHAR TO_DATE
TO_JSON TO_MILLIS TOLOWER TOUPPER TRANS_ARRAY TRANS_COLS
TRANSFORM TRANSFORM_KEYS TRANSFORM_VALUES TRANSLATE TRIM TRUNC
UNBASE64 UNHEX UNIQUE_ID UNIX_TIMESTAMP URL_DECODE URL_ENCODE
UUID VAR_SAMP VARIANCE/VAR_POP WEEKDAY WEEKOFYEAR WIDTH_BUCKET
WM_CONCAT YEAR ZIP_WITH
""".strip().split())


class ODPSEngineSpec(BaseEngineSpec):
    engine = "odps"
    engine_name = "ODPS"

    # pylint: disable=line-too-long
    _time_grain_expressions = {
        None: "{col}",
        TimeGrain.SECOND: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:ss')",
        TimeGrain.MINUTE: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:mm:00')",
        TimeGrain.HOUR: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd HH:00:00')",
        TimeGrain.DAY: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.WEEK: "date_format(date_sub({col}, CAST(7-from_unixtime(unix_timestamp({col}),'u') as int)), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.MONTH: "from_unixtime(unix_timestamp({col}), 'yyyy-MM-01 00:00:00')",
        TimeGrain.QUARTER: "date_format(add_months(datetrunc({col}, 'MM'), -(month({col})-1)%3), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.YEAR: "from_unixtime(unix_timestamp({col}), 'yyyy-01-01 00:00:00')",
        TimeGrain.WEEK_ENDING_SATURDAY: "date_format(date_add({col}, INT(6-from_unixtime(unix_timestamp({col}), 'u'))), 'yyyy-MM-dd 00:00:00')",
        TimeGrain.WEEK_STARTING_SUNDAY: "date_format(date_add({col}, -INT(from_unixtime(unix_timestamp({col}), 'u'))), 'yyyy-MM-dd 00:00:00')",
    }

    @classmethod
    def _get_odps_entry(cls, database):
        en = cls.get_engine(database)
        if hasattr(en, "__enter__"):
            engine = en.__enter__()
        else:
            engine = en

        odps_entry = engine.dialect.get_odps_from_url(engine.url)

        if hasattr(en, "__exit__"):
            en.__exit__(None, None, None)
        return odps_entry

    @classmethod
    def get_catalog_names(  # pylint: disable=unused-argument
        cls,
        database,
        inspector,
    ):
        engine = inspector.engine
        odps_entry = engine.dialect.get_odps_from_url(engine.url)
        try:
            return [proj.name for proj in odps_entry.list_projects()]
        except:
            return [odps_entry.project]

    @classmethod
    def where_latest_partition(  # pylint: disable=too-many-arguments
        cls,
        table_name,
        schema,
        database,
        query,
        columns=None,
    ):
        odps_entry = cls._get_odps_entry(database)
        table = odps_entry.get_table(table_name, schema=schema)
        if not table.schema.partitions:
            return None

        max_pt = table.get_max_partition()
        if columns is None or not max_pt:
            return None

        res_cols = set(c.get("name") for c in columns)
        for col_name, value in max_pt.partition_spec.items():
            if col_name in res_cols:
                query = query.where(Column(col_name) == value)
        return query

    @classmethod
    def latest_sub_partition(  # type: ignore
        cls, table_name, schema, database, **kwargs
    ):
        # TODO: implement`
        pass

    @classmethod
    def get_table_names(cls, database, inspector, schema):
        logger.info("Start listing tables for schema %s", schema)
        tables = super(ODPSEngineSpec, cls).get_table_names(
            database, inspector, schema
        )
        return set([
            n for n in tables if not n.startswith(TEMP_TABLE_PREFIX)
        ])

    @classmethod
    @cache_manager.cache.memoize()
    def get_function_names(cls, database):
        odps_entry = cls._get_odps_entry(database)
        funcs = set([
            func.name for func in odps_entry.list_functions()
            if not func.name.startswith("pyodps_")
        ])
        funcs = funcs | _builtin_funcs
        return sorted(funcs)

    @classmethod
    def execute(cls, cursor, query, **kwargs):
        options.verbose = True
        if not cls.allows_sql_comments:
            query = sql_parse.strip_comments_from_sql(query)

        if cls.arraysize:
            cursor.arraysize = cls.arraysize
        try:
            hints = {
                "odps.sql.jobconf.odps2": "true",
            }
            if not getattr(cursor.connection, "_project_as_schema", True):
                # sqlalchemy cursor need odps schema support
                hints.update({
                    "odps.sql.allow.namespace.schema": "true",
                    "odps.namespace.schema": "true",
                })
            cursor.execute(query, hints=hints)
        except Exception as ex:
            six.raise_from(cls.get_dbapi_mapped_exception(ex), ex)

    @classmethod
    def df_to_sql(cls, database, table, df, to_sql_kwargs):
        options.verbose = True
        odps_entry = cls._get_odps_entry(database)

        if to_sql_kwargs["if_exists"] == "fail":
            # Ensure table doesn't already exist.
            if odps_entry.exist_table(table.table, schema=table.schema):
                raise SupersetException("Table already exists")
        elif to_sql_kwargs["if_exists"] == "replace":
            odps_entry.delete_table(table.table, schema=table.schema, if_exists=True)

        odps_df = DataFrame(df)
        odps_df.persist(table.table, overwrite=False, odps=odps_entry)
