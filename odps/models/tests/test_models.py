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

import json

import pytest
from mock import MagicMock, Mock, patch

from ... import errors
from ... import types as odps_types
from ..ml.model import Model, ModelFieldSchema
from ..ml.models import Models
from ..ml.modelversions import ModelVersions


def test_model_field_schema_from_and_to_odps_schema():
    # Create an OdpsSchema with various column types
    original_schema = odps_types.OdpsSchema(
        [
            odps_types.Column("id", odps_types.bigint, nullable=False),
            odps_types.Column(
                "name", odps_types.string, comment="User name", nullable=True
            ),
            odps_types.Column(
                "amount",
                odps_types.Decimal(10, 2),
                comment="Transaction amount",
                nullable=False,
            ),
            odps_types.Column(
                "created_at",
                odps_types.datetime,
                comment="Creation timestamp",
                nullable=True,
            ),
        ]
    )

    # Convert OdpsSchema to ModelFieldSchema
    model_field_schema = ModelFieldSchema.from_odps_schema(original_schema)

    # Verify the ModelFieldSchema structure
    assert model_field_schema.fields is not None
    assert len(model_field_schema.fields) == 4

    # Check individual fields
    id_field = model_field_schema.fields[0]
    assert id_field.field_name == "id"
    assert id_field.type_definition == "BIGINT"
    assert id_field.type_category == "Bigint"
    assert id_field.mode == ModelFieldSchema.ModelFieldSchemaMode.REQUIRED
    assert id_field.description is None

    name_field = model_field_schema.fields[1]
    assert name_field.field_name == "name"
    assert name_field.type_definition == "STRING"
    assert name_field.type_category == "String"
    assert name_field.mode == ModelFieldSchema.ModelFieldSchemaMode.NULLABLE
    assert name_field.description == "User name"

    amount_field = model_field_schema.fields[2]
    assert amount_field.field_name == "amount"
    assert amount_field.type_definition == "DECIMAL(10,2)"
    assert amount_field.type_category == "Decimal"
    assert amount_field.mode == ModelFieldSchema.ModelFieldSchemaMode.REQUIRED
    assert amount_field.description == "Transaction amount"
    assert amount_field.precision == 10
    assert amount_field.scale == 2

    datetime_field = model_field_schema.fields[3]
    assert datetime_field.field_name == "created_at"
    assert datetime_field.type_definition == "DATETIME"
    assert datetime_field.type_category == "Datetime"
    assert datetime_field.mode == ModelFieldSchema.ModelFieldSchemaMode.NULLABLE
    assert datetime_field.description == "Creation timestamp"

    # Convert back to OdpsSchema
    converted_schema = model_field_schema.to_odps_schema()

    # Verify the round-trip conversion
    assert len(converted_schema.columns) == len(original_schema.columns)

    for original_col, converted_col in zip(
        original_schema.columns, converted_schema.columns
    ):
        assert original_col.name == converted_col.name
        assert str(original_col.type) == str(converted_col.type)
        assert original_col.comment == converted_col.comment
        assert original_col.nullable == converted_col.nullable


@pytest.fixture
def mock_client():
    return Mock(endpoint="mock_endpoint")


@pytest.fixture
def models_container(mock_client):
    return Models(client=mock_client)


def test_models_iteration_single_page(mock_client, models_container):
    # Mock response for single page iteration
    mock_response = Mock()
    mock_response.text = json.dumps(
        {"models": [{"name": "model1"}, {"name": "model2"}], "nextPageToken": None}
    )

    with patch("odps.models.ml.models.Models.parse") as mock_parse:
        mock_parsed = Mock()
        mock_parsed.models = [
            Model(client=mock_client, name="model1"),
            Model(client=mock_client, name="model2"),
        ]
        mock_parsed.next_page_token = None
        mock_parse.return_value = mock_parsed

        mock_client.get.return_value = mock_response

        # Test iteration
        models_list = list(models_container.iterate())

        # Assertions
        assert len(models_list) == 2
        assert models_list[0].name == "model1"
        assert models_list[1].name == "model2"
        mock_client.get.assert_called_once()


def test_models_iteration_multiple_pages(mock_client):
    # Create a more realistic parent structure for Models
    parent = MagicMock()
    parent.__getitem__.side_effects = lambda self, name: name
    parent.resource.return_value = "project/schemas/schema/models/model_name"
    models_container = Models(client=mock_client, parent=parent)

    # Mock responses for multiple page iteration
    response1 = Mock()
    response1.text = json.dumps(
        {"models": [{"name": "model1"}, {"name": "model2"}], "nextPageToken": "token1"}
    )

    response2 = Mock()
    response2.text = json.dumps(
        {"models": [{"name": "model3"}, {"name": "model4"}], "nextPageToken": None}
    )

    with patch("odps.models.ml.models.Models.parse") as mock_parse:
        # First call to parse
        mock_parsed1 = Mock()
        mock_parsed1.models = [
            Model(client=mock_client, name="model1"),
            Model(client=mock_client, name="model2"),
        ]
        mock_parsed1.next_page_token = "token1"

        # Second call to parse
        mock_parsed2 = Mock()
        mock_parsed2.models = [
            Model(client=mock_client, name="model3"),
            Model(client=mock_client, name="model4"),
        ]
        mock_parsed2.next_page_token = None

        mock_parse.side_effect = [mock_parsed1, mock_parsed2]
        mock_client.get.side_effect = [response1, response2]

        # Test iteration
        models_list = list(models_container.iterate())

        # Assertions
        assert len(models_list) == 4
        assert models_list[0].name == "model1"
        assert models_list[1].name == "model2"
        assert models_list[2].name == "model3"
        assert models_list[3].name == "model4"
        assert mock_client.get.call_count == 2


def test_models_contains_existing_model(mock_client, models_container):
    # Mock a successful reload (model exists)
    mock_model = Mock()
    mock_model.reload.return_value = None

    with patch.object(models_container, "_get", return_value=mock_model):
        # Test existence check
        result = "test_model" in models_container

        # Assertions
        assert result is True
        mock_model.reload.assert_called_once()


def test_models_contains_non_existing_model(mock_client, models_container):
    # Mock a NoSuchObject exception (model doesn't exist)
    mock_model = Mock()
    mock_model.reload.side_effect = errors.NoSuchObject("Model not found")

    with patch.object(models_container, "_get", return_value=mock_model):
        assert "nonexistent_model" not in models_container
        mock_model.reload.assert_called_once()

    # Test with invalid item type
    result = 123 in models_container
    assert result is False


@pytest.fixture
def mock_parent():
    parent = Mock()
    parent.resource.return_value = "/test/resource"
    return parent


@pytest.fixture
def model_versions_container(mock_client, mock_parent):
    container = ModelVersions(client=mock_client, parent=mock_parent)
    container._model_name = "test_model"
    return container


def test_model_versions_iteration_single_page(
    mock_client, mock_parent, model_versions_container
):
    # Mock response for single page iteration
    mock_response = Mock()
    mock_response.text = json.dumps(
        {
            "models": [
                {"name": "test_model", "version": "v1"},
                {"name": "test_model", "version": "v2"},
            ],
            "nextPageToken": None,
        }
    )

    with patch("odps.models.ml.modelversions.ModelVersions.parse") as mock_parse:
        mock_parsed = Mock()
        mock_parsed.models = [
            Model(client=mock_client, name="test_model", version_name="v1"),
            Model(client=mock_client, name="test_model", version_name="v2"),
        ]
        mock_parsed.next_page_token = None
        mock_parse.return_value = mock_parsed

        mock_client.get.return_value = mock_response

        # Test iteration
        models_list = list(model_versions_container.iterate())

        # Assertions
        assert len(models_list) == 2
        assert models_list[0].version_name == "v1"
        assert models_list[1].version_name == "v2"

        # Verify the correct URL was called
        mock_client.get.assert_called_once()
        args, kwargs = mock_client.get.call_args
        assert ":listVersions" in args[0]


def test_model_versions_contains_existing_model(mock_client, model_versions_container):
    # Mock a successful reload (model version exists)
    mock_model = Mock()
    mock_model.reload.return_value = None

    with patch.object(model_versions_container, "_get", return_value=mock_model):
        # Test existence check
        result = "v1" in model_versions_container

        # Assertions
        assert result is True
        mock_model.reload.assert_called_once()


def test_model_versions_contains_non_existing_model(
    mock_client, model_versions_container
):
    # Mock a NoSuchObject exception (model version doesn't exist)
    mock_model = Mock()
    mock_model.reload.side_effect = errors.NoSuchObject("Model version not found")

    with patch.object(model_versions_container, "_get", return_value=mock_model):
        # Test existence check
        assert "nonexistent_version" not in model_versions_container
        mock_model.reload.assert_called_once()


def test_model_versions_contains_model_object(mock_client, model_versions_container):
    # Test with Model object directly
    mock_model = Mock(spec=Model)
    mock_model.reload.return_value = None

    # Test existence check
    assert mock_model in model_versions_container
    # Should not call _get or reload since we're passing a Model object directly
    mock_model.reload.assert_called_once()
