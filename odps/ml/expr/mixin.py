# encoding: utf-8
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import collections
import itertools
import uuid

from . import op
from .core import AlgoExprMixin
from ..enums import FieldRole
from ..utils import KVConfig, MLField
from ...compat import six, reduce, OrderedDict
from ...df import DataFrame
from ...df.backends.odpssql.types import odps_schema_to_df_schema, df_schema_to_odps_schema, \
    df_type_to_odps_type
from ...df.expr.collections import Node, CollectionExpr, SequenceExpr, Column
from ...df.expr.dynamic import DynamicMixin, DynamicCollectionExpr, DynamicSchema
from ...models.table import TableSchema


class DynamicDataFrame(DynamicMixin, DataFrame):
    def __init__(self, *args, **kw):
        DynamicMixin.__init__(self)
        DataFrame.__init__(self, *args, **kw)

    _project = DynamicCollectionExpr._project


def copy_df(df):
    if isinstance(df, AlgoExprMixin):
        new_df = df.copy(register_expr=True)
    else:
        new_df = df.copy()
    new_df._ml_fields = df._ml_fields
    new_df._ml_uplink = [df]
    return new_df


def _get_field_name(field):
    if isinstance(field, SequenceExpr):
        return field.name
    else:
        return field


def _render_field_set(fields):
    if isinstance(fields, six.string_types) or not isinstance(fields, collections.Iterable):
        fields = [fields, ]
    fields = [f.name if isinstance(f, SequenceExpr) else f for f in fields]
    field_arrays = map(lambda v: v.replace(',', ' ').split(' ') if isinstance(v, six.string_types) else v, fields)
    return reduce(lambda a, b: set(a) | set(b), field_arrays, set()) - set(['', ])


def _change_singleton_roles(df, role_map, clear_feature):
    new_df = copy_df(df)
    new_df._perform_operation(op.SingletonRoleOperation(role_map, clear_feature))
    return new_df


def _batch_change_roles(df, fields, role, augment):
    new_df = copy_df(df)
    new_df._perform_operation(op.BatchRoleOperation(fields, role, augment))
    return new_df


class MLCollectionMixin(Node):
    """
    PyODPS ML Plugin for odps.df.core.CollectionExpr. This plugin is installed automatically,
    you do not need to instantiate this class manually.
    """
    @property
    def _ml_fields(self):
        if getattr(self, '_ml_fields_cache', None) is None:
            fields = OrderedDict((col.name, MLField.from_column(col))
                                 for col in df_schema_to_odps_schema(self.schema))
            to_be_fixed = dict((k, k) for k in six.iterkeys(fields))

            for n in self.traverse(top_down=True, unique=True,
                                   stop_cond=lambda it: getattr(it, '_ml_fields_cache', None) is not None):
                if isinstance(n, Column):
                    if n.source_name != n.name and n.name in to_be_fixed:
                        to_be_fixed[n.source_name] = to_be_fixed[n.name]

            for n in self.traverse(top_down=True, unique=True,
                                   stop_cond=lambda it: getattr(it, '_ml_fields_cache', None) is not None):
                if isinstance(n, CollectionExpr) and getattr(n, '_ml_fields_cache', None) is not None:
                    for col in n.columns:
                        if col.name not in to_be_fixed:
                            continue
                        field_obj = [f for f in n._ml_fields_cache if f.name == col.name]
                        fields[to_be_fixed[col.name]].role = field_obj[0].role
            for f in six.itervalues(fields):
                if not f.role:
                    f.role = set([FieldRole.FEATURE])
            self._ml_fields_cache = list(six.itervalues(fields))
        return self._ml_fields_cache

    @_ml_fields.setter
    def _ml_fields(self, value):
        self._ml_fields_cache = value

    def _rebuild_df_schema(self, dynamic=False):
        self._schema = odps_schema_to_df_schema(
            TableSchema.from_dict(OrderedDict([(f.name, f.type) for f in self._ml_fields]))
        )
        if dynamic:
            self._schema = DynamicSchema.from_schema(self._schema)

    def _perform_operation(self, op):
        op.execute(self._ml_uplink, self)
        self._ml_operations.append(op)

    def _assert_ml_fields_valid(self, *fields):
        if any(f not in self.schema for f in fields):
            raise ValueError('Column not found in DataFrame.')

    ###################
    # meta operations #
    ###################
    def exclude_fields(self, *args):
        """
        Exclude one or more fields from feature fields.

        :rtype: DataFrame
        """
        if not args:
            raise ValueError("Field list cannot be None.")
        new_df = copy_df(self)
        fields = _render_field_set(args)
        self._assert_ml_fields_valid(*fields)
        new_df._perform_operation(op.ExcludeFieldsOperation(fields))
        return new_df

    def select_features(self, *args, **kwargs):
        """
        Select one or more fields as feature fields.

        :rtype: DataFrame
        """
        if not args:
            raise ValueError("Field list cannot be empty.")
        # generate selected set from args
        augment = kwargs.get('add', False)
        fields = _render_field_set(args)
        self._assert_ml_fields_valid(*fields)
        return _batch_change_roles(self, fields, FieldRole.FEATURE, augment)

    def weight_field(self, f):
        """
        Select one field as the weight field.

        Note that this field will be exclude from feature fields.

        :param f: Selected weight field
        :type f: str
        :rtype: DataFrame
        """
        if f is None:
            raise ValueError("Field name cannot be None.")
        self._assert_ml_fields_valid(f)
        return _change_singleton_roles(self, {f: FieldRole.WEIGHT}, clear_feature=True)

    def label_field(self, f):
        """
        Select one field as the label field.

        Note that this field will be exclude from feature fields.

        :param f: Selected label field
        :type f: str
        :rtype: DataFrame
        """
        if f is None:
            raise ValueError("Label field name cannot be None.")
        self._assert_ml_fields_valid(f)
        return _change_singleton_roles(self, {_get_field_name(f): FieldRole.LABEL}, clear_feature=True)

    def continuous(self, *args):
        """
        Set fields to be continuous.

        :rtype: DataFrame

        :Example:

        >>> # Table schema is create table test(f1 double, f2 string)
        >>> # Original continuity: f1=DISCRETE, f2=DISCRETE
        >>> # Now we want to set ``f1`` and ``f2`` into continuous
        >>> new_ds = df.continuous('f1 f2')
        """
        new_df = copy_df(self)
        fields = _render_field_set(args)
        self._assert_ml_fields_valid(*fields)
        new_df._perform_operation(op.FieldContinuityOperation(dict((_get_field_name(f), True) for f in fields)))
        return new_df

    def discrete(self, *args):
        """
        Set fields to be discrete.

        :rtype: DataFrame

        :Example:

        >>> # Table schema is create table test(f1 double, f2 string)
        >>> # Original continuity: f1=CONTINUOUS, f2=CONTINUOUS
        >>> # Now we want to set ``f1`` and ``f2`` into continuous
        >>> new_ds = df.discrete('f1 f2')
        """
        new_df = copy_df(self)
        fields = _render_field_set(args)
        self._assert_ml_fields_valid(*fields)
        new_df._perform_operation(op.FieldContinuityOperation(dict((_get_field_name(f), False) for f in fields)))
        return new_df

    def key_value(self, *args, **kwargs):
        """
        Set fields to be key-value represented.

        :rtype: DataFrame

        :Example:

        >>> new_ds = df.key_value('f1 f2', kv=':', item=',')
        """
        new_df = copy_df(self)
        fields = _render_field_set(args)
        self._assert_ml_fields_valid(*fields)
        new_df._perform_operation(
            op.FieldKVConfigOperation(dict((_get_field_name(f), KVConfig(**kwargs)) for f in fields)))
        return new_df

    def erase_key_value(self, *args):
        """
        Erase key-value represented fields.

        :rtype: DataFrame

        :Example:

        >>> new_ds = df.erase_key_value('f1 f2')
        """
        new_df = copy_df(self)
        fields = _render_field_set(args)
        self._assert_ml_fields_valid(*fields)
        new_df._perform_operation(op.FieldKVConfigOperation(dict((_get_field_name(f), None) for f in fields)))
        return new_df

    def roles(self, clear_features=True, **field_roles):
        """
        Set roles of fields

        :param clear_features: Clear feature roles on fields
        :param field_roles:
        :return:
        """
        field_roles = dict((k, v.name if isinstance(v, SequenceExpr) else v)
                           for k, v in six.iteritems(field_roles))
        self._assert_ml_fields_valid(*list(six.itervalues(field_roles)))
        field_roles = dict((_get_field_name(f), MLField.translate_role_name(role))
                           for role, f in six.iteritems(field_roles))
        if field_roles:
            return _change_singleton_roles(self, field_roles, clear_features)
        else:
            return self

    ##########################
    # simple transformations #
    ##########################
    def split(self, frac):
        """
        Split the DataFrame into two DataFrames with certain ratio.

        :param frac: Split ratio
        :type frac: float

        :return: two split DataFrame objects
        :rtype: list[DataFrame]
        """
        from .. import preprocess
        split_obj = getattr(preprocess, '_Split')(fraction=frac)
        return split_obj.transform(self)

    def append_id(self, id_col_name='append_id', cols=None):
        """
        Append an ID column to current DataFrame.

        :param str id_col_name: name of appended ID field.
        :param str cols: fields contained in output. All fields by default.

        :return: DataFrame with ID field
        :rtype: DataFrame
        """
        from .. import preprocess
        if id_col_name in self.schema:
            raise ValueError('ID column collides with existing columns.')
        append_id_obj = getattr(preprocess, '_AppendID')(id_col=id_col_name, selected_cols=cols)
        return append_id_obj.transform(self)

    def merge_with(self, *dfs, **kwargs):
        return merge_data(self, *dfs, **kwargs)

    def _xflow_sample(self, columns=None, n=None, frac=None, replace=False, weights=None, strata=None,
                      random_state=None):
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and all(isinstance(df, pd.DataFrame) for df in self.data_source()):
            sample_func = getattr(self, '__sample')
            return sample_func(columns=columns, n=n, frac=frac, replace=replace,
                               weights=weights, strata=strata, random_state=random_state)
        from .. import preprocess

        if weights is not None:
            if not isinstance(weights, (six.string_types, SequenceExpr)):
                raise ValueError('weights should be the name of the weight column.')
            algo_cls = getattr(preprocess, '_WeightedSample')
            algo_obj = algo_cls(sample_size=n, sample_ratio=frac, prob_col=_get_field_name(weights),
                                replace=replace, random_seed=random_state)
        elif strata is not None:
            def dict_to_kv(d):
                if not isinstance(d, dict):
                    return d
                return ','.join('{0}:{1}'.format(k, v) for k, v in six.iteritems(d))

            if replace:
                raise ValueError('Stratified sampling with replacement is not supported.')
            algo_cls = getattr(preprocess, '_StratifiedSample')
            algo_obj = algo_cls(sample_size=dict_to_kv(n), sample_ratio=dict_to_kv(frac),
                                strata_col_name=_get_field_name(strata), random_seed=random_state)
        else:
            algo_cls = getattr(preprocess, '_RandomSample')
            algo_obj = algo_cls(sample_size=n, sample_ratio=frac, replace=replace,
                                random_seed=random_state)
        return algo_obj.transform(self)


class MLSequenceMixin(Node):
    def _perform_operation(self, op):
        op.execute(self._ml_uplink, self)
        self._ml_operations.append(op)

    @property
    def _ml_fields(self):
        if getattr(self, '_ml_fields_cache', None) is None:
            if hasattr(self, 'input'):
                for fobj in self.input._ml_fields:
                    if fobj.name == self.name:
                        self._ml_fields_cache = [fobj.copy()]
                        break
            self._ml_fields_cache = [MLField(self.name, df_type_to_odps_type(self.dtype).name, FieldRole.FEATURE)]
        return self._ml_fields_cache

    @_ml_fields.setter
    def _ml_fields(self, value):
        self._ml_fields_cache = value

    def continuous(self):
        """
        Set sequence to be continuous.

        :rtype: Column

        :Example:

        >>> # Table schema is create table test(f1 double, f2 string)
        >>> # Original continuity: f1=DISCRETE, f2=DISCRETE
        >>> # Now we want to set ``f1`` and ``f2`` into continuous
        >>> new_ds = df.continuous('f1 f2')
        """
        field_name = self.name
        new_df = copy_df(self)
        new_df._perform_operation(op.FieldContinuityOperation({field_name: True}))
        return new_df

    def discrete(self):
        """
        Set sequence to be discrete.

        :rtype: Column

        :Example:

        >>> # Table schema is create table test(f1 double, f2 string)
        >>> # Original continuity: f1=CONTINUOUS, f2=CONTINUOUS
        >>> # Now we want to set ``f1`` and ``f2`` into continuous
        >>> new_ds = df.discrete('f1 f2')
        """
        field_name = self.name
        new_df = copy_df(self)
        new_df._perform_operation(op.FieldContinuityOperation({field_name: False}))
        return new_df

    def key_value(self, **kwargs):
        """
        Set fields to be key-value represented.

        :rtype: Column

        :Example:

        >>> new_ds = df.key_value('f1 f2', kv=':', item=',')
        """
        field_name = self.name
        new_df = copy_df(self)
        new_df._perform_operation(op.FieldKVConfigOperation({field_name: KVConfig(**kwargs)}))
        return new_df

    def erase_key_value(self):
        """
        Erase key-value represented fields.

        :rtype: Column

        :Example:

        >>> new_ds = df.erase_key_value('f1 f2')
        """
        field_name = self.name
        new_df = copy_df(self)
        new_df._perform_operation(op.FieldKVConfigOperation({field_name: None}))
        return new_df

    def role(self, role_name):
        """
        Set role of current column

        :param role_name: name of the role to be selected.
        :return:
        """
        field_name = self.name
        field_roles = {field_name: MLField.translate_role_name(role_name)}
        if field_roles:
            return _change_singleton_roles(self, field_roles, True)
        else:
            return self

    ##########################
    # simple transformations #
    ##########################
    def merge_with(self, *dfs, **kwargs):
        return merge_data(self, *dfs, **kwargs)


def _xflow_split(expr, frac, seed=None):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and all(isinstance(df, pd.DataFrame) for df in expr.data_source()):
        split_func = getattr(expr, '_split')
        return split_func(frac, seed=seed)

    from .. import preprocess
    split_obj = getattr(preprocess, '_Split')(fraction=frac, random_seed=seed)
    return split_obj.transform(expr)


def _xflow_append_id(expr, id_col='append_id'):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and all(isinstance(df, pd.DataFrame) for df in expr.data_source()):
        append_id_func = getattr(expr, '_append_id')
        return append_id_func(id_col)

    from .. import preprocess
    append_id_obj = getattr(preprocess, '_AppendID')(id_col=id_col)
    return append_id_obj.transform(expr)


def _xflow_concat(left, rights):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    chained = itertools.chain((left, ), rights)
    if pd is not None and all(isinstance(df, pd.DataFrame) for i in chained for df in i.data_source()):
        concat_func = getattr(left, '__horz_concat')
        return concat_func(rights)

    return merge_data(left, *rights)

MLCollectionMixin._xflow_concat = _xflow_concat
MLCollectionMixin._xflow_append_id = _xflow_append_id
MLCollectionMixin._xflow_split = _xflow_split


def ml_collection_mixin(cls):
    def setter_generator(role, clear_feature=True):
        def role_setter(self, field_name):
            if field_name is None:
                raise ValueError("Field name cannot be None.")
            if isinstance(field_name, six.string_types):
                return _change_singleton_roles(self, {field_name: role}, clear_feature=clear_feature)
            else:
                return _batch_change_roles(self, field_name, role, False)
        return role_setter

    if hasattr(cls, 'field_role_enum'):
        field_role_enum = cls.field_role_enum
        delattr(cls, 'field_role_enum')

        MLField.register_roles(field_role_enum)
        if hasattr(cls, 'non_feature_roles'):
            non_feature_roles = set(cls.non_feature_roles)
            delattr(cls, 'non_feature_roles')
        else:
            non_feature_roles = set()
        for role in field_role_enum:
            setter = setter_generator(role, role in non_feature_roles)
            setter.__name__ = '%s_field' % role.name.lower()
            setattr(cls, setter.__name__, setter)

    if cls not in MLCollectionMixin.__bases__:
        MLCollectionMixin.__bases__ += (cls, )
    return cls


def merge_data(*data_frames, **kwargs):
    """
    Merge DataFrames by column. Number of rows in tables must be the same.

    This method can be called both outside and as a DataFrame method.

    :param list[DataFrame] data_frames: DataFrames to be merged.
    :param bool auto_rename: if True, fields in source DataFrames will be renamed in the output.

    :return: merged data frame.
    :rtype: DataFrame

    :Example:
    >>> merged1 = merge_data(df1, df2)
    >>> merged2 = df1.merge_with(df2, auto_rename_col=True)
    """
    from .specialized import build_merge_expr
    from ..utils import ML_ARG_PREFIX

    if len(data_frames) <= 1:
        raise ValueError('Count of DataFrames should be at least 2.')

    norm_data_pairs = []
    df_tuple = collections.namedtuple('MergeTuple', 'df cols exclude')
    for pair in data_frames:
        if isinstance(pair, tuple):
            if len(pair) == 2:
                df, cols = pair
                exclude = False
            else:
                df, cols, exclude = pair
            if isinstance(cols, six.string_types):
                cols = cols.split(',')
        else:
            df, cols, exclude = pair, None, False
        norm_data_pairs.append(df_tuple(df, cols, exclude))

    auto_rename = kwargs.get('auto_rename', False)

    sel_cols_dict = dict((idx, tp.cols) for idx, tp in enumerate(norm_data_pairs) if tp.cols and not tp.exclude)
    ex_cols_dict = dict((idx, tp.cols) for idx, tp in enumerate(norm_data_pairs) if tp.cols and tp.exclude)

    merge_expr = build_merge_expr(len(norm_data_pairs))
    arg_dict = dict(_params={'autoRenameCol': str(auto_rename)},
                    selected_cols=sel_cols_dict, excluded_cols=ex_cols_dict)
    for idx, dp in enumerate(norm_data_pairs):
        arg_dict[ML_ARG_PREFIX + 'input%d' % (1 + idx)] = dp.df

    out_df = merge_expr(register_expr=True, _exec_id=uuid.uuid4(), _output_name='output', **arg_dict)

    out_df._ml_uplink = [dp.df for dp in norm_data_pairs]
    out_df._perform_operation(op.MergeFieldsOperation(auto_rename, sel_cols_dict, ex_cols_dict))
    out_df._rebuild_df_schema()
    return out_df


class MLSchema(TableSchema):
    __slots__ = '_collection',

    class MLAttrCollection(object):
        def __init__(self, collection=None):
            if collection is not None:
                self._ml_fields = dict((f.name, f) for f in collection._ml_fields)
            else:
                self._ml_fields = dict()

        def __getitem__(self, item):
            return self._ml_fields.get(item)

        def __contains__(self, item):
            return item in self._ml_fields

    @property
    def ml_attr(self):
        if hasattr(self, '_collection'):
            return self.MLAttrCollection(self._collection)
        else:
            return self.MLAttrCollection()

    def _repr(self):
        original_repr = super(MLSchema, self)._repr()
        sio = six.StringIO()
        ml_attrs = self.ml_attr
        for line in original_repr.splitlines():
            if not line.startswith(' '):
                sio.write(line + '\n')
                continue
            ftypes = [s.strip() for s in line.strip().rsplit(' ', 1)]
            if ftypes[0].startswith('"') and ftypes[0].endswith('"'):
                if isinstance(ftypes[0], bytes):
                    ftypes[0] = ftypes[0][1:-1].decode('unicode-escape').encode('utf-8')
                else:
                    ftypes[0] = ftypes[0][1:-1].decode('unicode-escape')

            if ftypes[0] not in ml_attrs:
                sio.write(line + '\n')
                continue

            ml_attr = ml_attrs[ftypes[0]]
            if ml_attr.kv_config:
                kv_repr = ml_attr._repr_type_()
                slen = len(line) - line.find(ftypes[1])
                if len(kv_repr) > slen:
                    line = line.replace(ftypes[1] + ' ' * (slen - len(ftypes[1])), kv_repr)
                elif len(kv_repr) > len(ftypes[1]):
                    line = line.replace(ftypes[1] + ' ' * (len(kv_repr) - len(ftypes[1])), kv_repr)
                else:
                    line = line.replace(ftypes[1], kv_repr + ' ' * (len(ftypes[1]) - len(kv_repr)))
            [sio.write(s) for s in (line, ' ', ml_attr._repr_role_(), '\n')]
        return sio.getvalue()


def collection_dtypes_wrapper(self):
    if getattr(self, '_ml_fields_cache', None) is None:
        return self.schema
    else:
        schema = MLSchema(columns=self.schema.columns, partitions=self.schema.partitions)
        schema._collection = self
        return schema


def install_mixin():
    CollectionExpr.__bases__ += (MLCollectionMixin, )
    SequenceExpr.__bases__ += (MLSequenceMixin, )
    CollectionExpr.dtypes = property(fget=collection_dtypes_wrapper)


install_mixin()
