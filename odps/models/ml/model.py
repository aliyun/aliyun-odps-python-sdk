# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from ... import serializers, types, utils
from ...compat import Enum
from ..core import JSONLazyLoad


def _parse_datetime(time_str):
    return utils.to_datetime(int(time_str))


class ModelFieldSchema(serializers.JSONSerializableModel):
    class ModelFieldSchemaMode(Enum):
        REQUIRED = "REQUIRED"
        NULLABLE = "NULLABLE"

    field_name = serializers.JSONNodeField("fieldName")
    type_definition = serializers.JSONNodeField(
        "sqlTypeDefinition", parse_callback=types.validate_data_type
    )
    type_category = serializers.JSONNodeField("typeCategory")
    mode = serializers.JSONNodeField(
        "mode",
        parse_callback=lambda s: ModelFieldSchema.ModelFieldSchemaMode(s.upper())
        if s is not None
        else None,
    )
    fields = serializers.JSONNodesReferencesField("ModelFieldSchema", "fields")
    description = serializers.JSONNodeField("description")
    max_length = serializers.JSONNodeField("maxLength", parse_callback=int)
    precision = serializers.JSONNodeField("precision", parse_callback=int)
    scale = serializers.JSONNodeField("scale", parse_callback=int)
    default_value_expression = serializers.JSONNodeField("defaultValueExpression")

    def to_odps_column(self):
        assert self.field_name is not None
        return types.Column(
            self.field_name,
            self.type_definition,
            comment=self.description,
            nullable=self.mode == ModelFieldSchema.ModelFieldSchemaMode.NULLABLE,
            generate_expression=self.default_value_expression,
        )

    def to_odps_schema(self):
        # only converts collections
        assert self.field_name is None
        cols = [f.to_odps_column() for f in self.fields or ()]
        return types.OdpsSchema(cols)

    @classmethod
    def from_odps_column(cls, odps_column):
        if isinstance(odps_column.type, (types.Array, types.Map, types.Struct)):
            raise NotImplementedError(
                "Cannot support column type %s in models" % odps_column.type
            )
        schema = ModelFieldSchema(
            field_name=odps_column.name,
            type_definition=str(odps_column.type),
            type_category=type(odps_column.type).__name__,
            mode=ModelFieldSchema.ModelFieldSchemaMode.NULLABLE
            if odps_column.nullable
            else ModelFieldSchema.ModelFieldSchemaMode.REQUIRED,
            description=odps_column.comment,
            default_value_expression=str(odps_column._generate_expression),
        )
        if isinstance(odps_column.type, types.Decimal):
            schema.precision = odps_column.type.precision
            schema.scale = odps_column.type.scale
        return schema

    @classmethod
    def from_odps_schema(cls, odps_schema):
        fields = [
            ModelFieldSchema.from_odps_column(odps_column)
            for odps_column in odps_schema.columns
        ]
        return ModelFieldSchema(fields=fields)


class Model(JSONLazyLoad):
    class ModelSourceType(Enum):
        IMPORT = "IMPORT"
        INTERNAL_TRAIN = "INTERNAL_TRAIN"
        REMOTE = "REMOTE"

    class ModelType(Enum):
        LLM = "LLM"
        MLLM = "MLLM"
        BOOSTED_TREE_CLASSIFIER = "BOOSTED_TREE_CLASSIFIER"
        BOOSTED_TREE_REGRESSOR = "BOOSTED_TREE_REGRESSOR"

    name = serializers.JSONNodeField("modelName")
    version_name = serializers.JSONNodeField("versionName", default=None)
    default_version = serializers.JSONNodeField("defaultVersion", default=None)
    creation_time = serializers.JSONNodeField(
        "createTime", parse_callback=_parse_datetime
    )
    last_modified_time = serializers.JSONNodeField(
        "updateTime", parse_callback=_parse_datetime
    )
    version_creation_time = serializers.JSONNodeField(
        "versionCreateTime", parse_callback=_parse_datetime
    )
    version_last_modified_time = serializers.JSONNodeField(
        "versionUpdateTime", parse_callback=_parse_datetime
    )
    description = serializers.JSONNodeField("description", default=None)
    expiration_days = serializers.JSONNodeField("expirationDays", default=None)
    version_expiration_days = serializers.JSONNodeField(
        "versionExpirationDays", default=None
    )
    source_type = serializers.JSONNodeField(
        "sourceType",
        parse_callback=lambda s: Model.ModelSourceType(s.upper())
        if s is not None
        else None,
    )
    type = serializers.JSONNodeField(
        "modelType",
        parse_callback=lambda s: Model.ModelType(s.upper()) if s is not None else None,
    )
    _labels = serializers.JSONNodeField("labels")
    path = serializers.JSONNodeField("path", default=None)
    options = serializers.JSONNodeField("options", default=None)
    training_info = serializers.JSONNodeField("trainingInfo", default=None)
    _feature_columns = serializers.JSONNodeReferenceField(
        ModelFieldSchema, "featureColumns", default=None
    )

    def __repr__(self):
        return "<Model %s version_name=%s>" % (self.name, self.version_name)

    @property
    def versions(self):
        from .modelversions import ModelVersions

        return ModelVersions(_parent=self, client=self._client, _model_name=self.name)

    @property
    def model_schema(self):
        if not self._feature_columns:
            return None
        return self._feature_columns.to_odps_schema()

    @model_schema.setter
    def model_schema(self, schema):
        if schema is None:
            self._feature_columns = None
        elif not isinstance(schema, types.OdpsSchema):
            raise TypeError("schema should be odps.types.OdpsSchema")
        self._feature_columns = ModelFieldSchema.from_odps_schema(schema)

    def reload(self):
        resp = self._client.get(self.resource(with_schema=True))
        self.parse(self._client, resp, obj=self)
