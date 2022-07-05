#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import uuid
import warnings

from .. import op, exporters
from ..core import AlgoExprMixin, AlgoCollectionExpr
from ...utils import MLField, FieldRole, FieldContinuity, import_class_member, build_model_table_name, ML_ARG_PREFIX
from ....compat import six
from ....config import options
from ....errors import NoSuchObject
from ....df.expr.collections import CollectionExpr, Expr
from ....df.expr.dynamic import DynamicMixin


_PREDICTION_ARGS = """
inputTableName modelName outputTableName featureColNames appendColNames inputTablePartitions
resultColName scoreColName detailColName enableSparse itemDelimiter kvDelimiter
""".split()
_PREDICTION_EXPORTERS = {
    "inputTableName": lambda e: exporters.get_input_table_name(e, "input"),
    "inputTablePartitions": lambda e: exporters.get_input_partitions(e, "input"),
    "modelName": lambda e: exporters.get_input_model_name(e, "model"),
    "outputTableName": lambda e: exporters.get_output_table_name(e, "output"),
    "outputTablePartition": lambda e: exporters.get_output_table_partitions(e, "output"),
    "appendColNames": lambda e: exporters.get_original_columns(e, "input"),
    "enableSparse": lambda e: exporters.get_enable_sparse(e, 'enableSparse', 'input'),
    "itemDelimiter": lambda e: exporters.get_item_delimiter(e, 'itemDelimiter', 'input'),
    "kvDelimiter": lambda e: exporters.get_kv_delimiter(e, 'kvDelimiter', 'input'),
    "featureColNames": lambda e: exporters.get_sparse_predict_feature_columns(e, 'featureColNames', 'input'),
}
_PREDICTION_META = {
    'xflowName': 'prediction',
    'xflowProjectName': 'algo_public',
}


class PredictionCollectionExpr(AlgoCollectionExpr):
    node_name = 'Prediction'
    _algo = 'prediction'
    _args = '_mlattr_input', '_mlattr_model'
    _exported = _PREDICTION_ARGS
    _exporters = _PREDICTION_EXPORTERS
    algo_meta = _PREDICTION_META

    def _init(self, *args, **kwargs):
        super(PredictionCollectionExpr, self)._init(*args, **kwargs)

        self._init_attr('_params', dict())
        self._init_attr('_engine_kw', dict())

    @property
    def input(self):
        return self._mlattr_input

    @property
    def model(self):
        return self._mlattr_model


class ModelDataCollectionExpr(CollectionExpr):
    node_name = 'ModelData'
    __slots__ = '_data_item',
    _args = '_mlattr_model',

    def _init(self, *args, **kwargs):
        kwargs.pop('register_expr', None)
        model_inputs = kwargs.pop('model_inputs', [])
        for mi in model_inputs:
            kwargs.pop(ML_ARG_PREFIX + mi, None)

        super(ModelDataCollectionExpr, self)._init(*args, **kwargs)

    @property
    def model(self):
        return self._mlattr_model

    def table_name(self, model_name=None):
        return self.model.table_name(self._data_item, model_name)

    def convert_params(self, src_expr=None):
        if src_expr is not None:
            src_expr = src_expr.model
        return self.model.convert_params(src_expr=src_expr)

    def accept(self, visitor):
        if self._source_data is not None:
            super(ModelDataCollectionExpr, self).accept(visitor)


class ODPSModelExpr(AlgoExprMixin, Expr):
    __slots__ = '_source_data', '_is_offline_model', '_model_collections', \
                '_model_params', '_predict_fields', '_predictor', '_recommender', '_pmml'
    _suffix = 'ModelExpr'
    _non_table = True
    _pmml_members = 'load_pmml', 'segments', 'regression'
    node_name = 'Model'

    def _fill_attr(self, attr, val):
        if not hasattr(self, attr):
            setattr(self, attr, val)
        if getattr(self, attr) is None:
            setattr(self, attr, val)

    def _init(self, *args, **kwargs):
        register_expr = kwargs.pop('register_expr', False)

        p_args = [a.cache() if isinstance(a, Expr) else a for a in args]
        p_kw = dict((k, self.cache_input(v)) for k, v in six.iteritems(kwargs))

        super(ODPSModelExpr, self)._init(*p_args, **p_kw)
        self.cache(False)

        self._fill_attr('_model_collections', dict())
        self._fill_attr('_model_params', dict())
        self._fill_attr('_is_offline_model', True)

        if hasattr(self, 'algo_meta'):
            for fun_name in ['predictor', 'recommender']:
                self._init_attr('_' + fun_name, self.algo_meta.get(fun_name, None))

        self._fill_attr('_predict_fields', [
            MLField('prediction_result', 'string', FieldRole.PREDICTED_CLASS, continuity=FieldContinuity.DISCRETE,
                    is_append=True),
            MLField('prediction_score', 'double', [FieldRole.PREDICTED_SCORE, FieldRole.PREDICTED_VALUE],
                    continuity=FieldContinuity.CONTINUOUS, is_append=True),
            MLField('prediction_detail', 'string', FieldRole.PREDICTED_DETAIL, is_append=True),
        ])

        if register_expr:
            self.register_expr()

    def _predict_offline_model(self, expr):
        df = PredictionCollectionExpr(register_expr=True, _mlattr_input=expr, _mlattr_model=self,
                                      _schema=expr.schema, _exec_id=uuid.uuid4(), _output_name='output')
        df._ml_uplink = [expr]

        append_op = op.StaticFieldChangeOperation(self._predict_fields, is_append=True)
        df._perform_operation(append_op)
        df._rebuild_df_schema(isinstance(expr, DynamicMixin))
        return df

    def _predict_tables_model(self, *args, **kwargs):
        from ...algolib.loader import dispatch_args
        method_path = kwargs.pop('_method_path', self._predictor)
        _predictor = import_class_member(method_path)
        members = set(p for p in dir(_predictor) if not p.startswith('_') and
                      not p.startswith('set_') and not callable(getattr(_predictor, p)))
        ml_args, p_args, ml_kw, p_kw = dispatch_args(members, *args, **kwargs)
        return _predictor(*p_args, **p_kw).transform(self, *ml_args, **ml_kw)

    def predict(self, *args, **kwargs):
        if self._is_offline_model:
            return self._predict_offline_model(*args, **kwargs)
        else:
            return self._predict_tables_model(*args, **kwargs)

    def _recommend(self, *args, **kwargs):
        kw = kwargs.copy()
        kw['_method_path'] = self._recommender
        return self._predict_tables_model(*args, **kw)

    def __getattr__(self, item):
        def raise_err():
            raise AttributeError("'{0}' object has no attribute '{1}'".format(type(self).__name__, item))

        if item.startswith('_'):
            raise_err()
        if item == 'recommend' and getattr(self, '_recommender', None) is not None:
            return self._recommend
        if item in type(self)._pmml_members:
            if not self._is_offline_model:
                raise_err()
            warnings.warn('Direct methods on Pmml Models is deprecated, see documentation '
                          'for more details.', DeprecationWarning)
            return getattr(self, '_' + item)
        if self._is_offline_model or item not in self._model_collections:
            raise_err()
        if item in self._model_collections:
            return self._model_collections[item]
        raise_err()

    def _data_source(self):
        if hasattr(self, '_source_data') and self._source_data is not None:
            yield self._source_data

    def table_name(self, item_name, model_name=None):
        model_name = model_name or str(self._source_data.name)
        return build_model_table_name(model_name, item_name)

    def accept(self, visitor):
        visitor.visit_algo(self)

    def get_cached(self, data):
        if self._is_offline_model:
            try:
                if not options.ml.dry_run:
                    data.reload()
            except NoSuchObject:
                from ....df.backends.context import context
                context.uncache(self)
                return None
            mod = type(self)(_source_data=data)
        else:
            mod = ODPSModelExpr(_source_data=data, _is_offline_model=False, _model_params=data.params.copy(),
                                _predictor=data.params.get('predictor'),
                                _recommender=data.params.get('recommender'))
            data_exprs = dict()
            for k, v in six.iteritems(data.tables):
                data_exprs[k] = ModelDataCollectionExpr(_mlattr_model=mod, _data_item=k)
                data_exprs[k]._source_data = v
            mod._model_collections = data_exprs
        mod._need_cache = False
        return mod

    def _load_pmml(self):
        if not getattr(self, '_pmml', None):
            pmml_obj = self.execute()
            self._pmml = pmml_obj.pmml
        return self._pmml

    @property
    def _regression(self):
        from .pmml import PmmlResult, PmmlRegressionResult
        self._load_pmml()
        result = PmmlResult(self._pmml)
        if not isinstance(result, PmmlRegressionResult):
            raise ValueError('`RegressionModel` element not found in PMML.')
        return result

    @property
    def _segments(self):
        from .pmml import PmmlResult, PmmlSegmentsResult
        self._load_pmml()
        result = PmmlResult(self._pmml)
        if not isinstance(result, PmmlSegmentsResult):
            raise ValueError('`Segmentation` element not found in PMML.')
        return result

    def persist(self, name, project=None, drop_model=False, **kwargs):
        """
        Persist the execution into a new model.

        :param name: model name
        :param project: name of the project
        :param drop_model: drop model before creation
        """
        return super(ODPSModelExpr, self).persist(name, project=project, drop_model=drop_model, **kwargs)

    def verify(self):
        return super(ODPSModelExpr, self).verify()


class PmmlModel(ODPSModelExpr):
    def _init(self, *args, **kwargs):
        if len(args) == 1:
            model = args[0]
        else:
            model = kwargs.pop('_source_data', None)
            if model is None:
                raise ValueError('ODPS offline model should be provided.')
        kwargs['_source_data'] = model
        kwargs['_is_offline_model'] = True
        kwargs['_exec_id'] = str(uuid.uuid4())

        super(PmmlModel, self)._init(*args, **kwargs)

        self.executed = True

    def predict(self, *args, **kwargs):
        """
        Predict given DataFrame using the given model. Actual prediction steps will not
        be executed till an operational step is called.

        After execution, three columns will be appended to the table:

        +-------------------+--------+----------------------------------------------------+
        | Field name        | Type   | Comments                                           |
        +===================+========+====================================================+
        | prediction_result | string | field indicating the predicted label, absent if    |
        |                   |        | the model is a regression model                    |
        +-------------------+--------+----------------------------------------------------+
        | prediction_score  | double | field indicating the score value if the model is   |
        |                   |        | a classification model, or the predicted value if  |
        |                   |        | the model is a regression model.                   |
        +-------------------+--------+----------------------------------------------------+
        | prediction_detail | string | field in JSON format indicating the score for      |
        |                   |        | every class.                                       |
        +-------------------+--------+----------------------------------------------------+

        :type df: DataFrame
        :rtype: DataFrame

        :Example:

        >>> model = PmmlModel(odps.get_offline_model('model_name'))
        >>> data = DataFrame(odps.get_table('table_name'))
        >>> # prediction below will not be executed till predicted.persist is called
        >>> predicted = model.predict(data)
        >>> predicted.persist('predicted')
        """
        return super(PmmlModel, self).predict(*args, **kwargs)


class TablesModel(ODPSModelExpr):
    def _init(self, *args, **kwargs):
        from ....df import DataFrame

        if len(args) == 1:
            model = args[0]
        else:
            model = kwargs.pop('_source_data', None)
            if model is None:
                raise ValueError('Name of the TablesModel should be provided.')

        model_collection = dict()
        for name, tb in six.iteritems(model.tables):
            model_collection[name] = DataFrame(tb)

        kwargs['_source_data'] = model
        kwargs['_is_offline_model'] = False
        kwargs['_model_collections'] = model_collection
        kwargs['_model_params'] = model.params

        for k in ('predictor', 'recommender'):
            kwargs['_' + k] = model.params.get(k)

        kwargs['_exec_id'] = str(uuid.uuid4())

        super(TablesModel, self)._init(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict given DataFrame using the given model. Actual prediction steps will not
        be executed till an operational step is called.

        :param list[DataFrame] args: input DataFrames to be predicted
        :param kwargs: named input DataFrames or prediction parameters, details can be found in ''Predictor Parameters'' section of training algorithms.

        A :class:`DataFrame` object will be generated for input data. Output fields may be referenced in the
        documents of training algorithms.
        """
        return super(TablesModel, self).predict(*args, **kwargs)


class TablesModelResult(object):
    def __init__(self, params, results):
        self._params = params
        self._results = results

    @property
    def params(self):
        return self._params

    @property
    def results(self):
        return self._results

    def __getattr__(self, item):
        if item in self._results:
            return self._results[item]
        raise AttributeError
