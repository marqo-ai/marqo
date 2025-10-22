import filecmp
import json
import os
import tarfile
import tempfile
import textwrap
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import cast
from unittest import mock
from unittest.mock import patch
import uuid

import httpx
import pytest

from marqo import version
from marqo.core.exceptions import IndexExistsError, ApplicationNotInitializedError, InternalError, \
    ApplicationRollbackError, OperationConflictError
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.index_management.vespa_application_package import (MarqoConfig, VespaApplicationPackage,
                                                                   ApplicationPackageDeploymentSessionStore)
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.typeahead.typeahead_vespa_schema import TypeaheadVespaSchema
from marqo.core.vespa_index.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.core.inference.embedding_models.marqo_model_regiestry import get_model_properties
from marqo.vespa.exceptions import VespaActivationConflictError
from marqo.vespa.models import VespaDocument
from tests.integ_tests.marqo_test import MarqoTestCase, TestImageUrls
from marqo.core.inference.api.exceptions import InferenceError


class TestIndexManagement(MarqoTestCase):
    """
    """
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        unstructured_image_index = cls.unstructured_marqo_index_request(
            model=Model(
                name='open_clip/ViT-B-32/laion2b_s34b_b79k',
                properties={
                    'name': 'open_clip/coca_ViT-B-32/laion2b_s13b_b90k',
                    'dimensions': 512,
                    'type': 'open_clip',
                },
                custom=True
            ),
            treat_urls_and_pointers_as_images=True
        )

        unstructured_text_index = cls.unstructured_marqo_index_request(
            model=Model(
                name='hf/all-MiniLM-L6-v2',
                properties={
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "tokens": 256,
                    "type": "hf",
                    "notes": ""
                },
                custom=True
            )
        )
        
        structured_image_index = cls.structured_marqo_index_request(
            model=Model(
                name='open_clip/ViT-B-32/laion2b_s34b_b79k',
                properties={
                    'name': 'open_clip/coca_ViT-B-32/laion2b_s13b_b90k',
                    'dimensions': 512,
                    'type': 'open_clip',
                },
                custom=True
            ),
            fields=[
                FieldRequest(name="image_field", type=FieldType.ImagePointer),
                FieldRequest(name="text_field", type=FieldType.Text)
            ],
            tensor_fields=["image_field", "text_field"]
        )

        structured_text_index = cls.structured_marqo_index_request(
            model=Model(
                name='hf/all-MiniLM-L6-v2',
                properties={
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "tokens": 256,
                    "type": "hf",
                    "notes": ""
                },
                custom=True
            ),
            fields=[
                FieldRequest(name="text_field", type=FieldType.Text)
            ],
            tensor_fields=["text_field"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_image_index,
            unstructured_text_index,
            structured_image_index,
            structured_text_index
        ])

        cls.unstructured_image_index = unstructured_image_index.name
        cls.unstructured_text_index = unstructured_text_index.name
        cls.structured_image_index = structured_image_index.name
        cls.structured_text_index = structured_text_index.name

    def setUp(self):
        super().setUp()
        self.index_management = IndexManagement(
            self.vespa_client,
            zookeeper_client=self.zookeeper_client,
            enable_index_operations=True,
            deployment_timeout_seconds=30,
            convergence_timeout_seconds=120
        )

    def test_updated_model_properties_for_unstructured_image_index(self):
        """
        Test that updating the model properties of an unstructured image index works correctly.
        """
        documents = [
            {
                "_id": "doc1",
                "image_field": TestImageUrls.IMAGE0.value,
                "text_field": "A sample text"
            }
        ]
        with self.assertRaises(InferenceError):
            self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    docs=documents, tensor_fields=["image_field", "text_field"], index_name=self.unstructured_image_index
                )
            )

        correct_model_properties = get_model_properties("open_clip/ViT-B-32/laion2b_s34b_b79k")
        self.index_management.update_index_settings_by_settings_dict(
            index_name=self.unstructured_image_index,
            settings_dict={"modelProperties": correct_model_properties}
        )

        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=documents, tensor_fields=["image_field", "text_field"], index_name=self.unstructured_image_index
            )
        )

        updated_index = self.index_management.get_index(self.unstructured_image_index)
        self.assertEqual(correct_model_properties, updated_index.model.properties)
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.unstructured_image_index).number_of_documents)
        self.assertEqual(2, self.monitoring.get_index_stats_by_name(self.unstructured_text_index).number_of_vectors)

    def test_updated_model_properties_for_unstructured_text_index(self):
        """
        Test that updating the model properties of an unstructured image index works correctly.
        """
        documents = [
            {
                "_id": "doc1",
                "text_field": "A sample text"
            }
        ]
        with self.assertRaises(InferenceError):
            self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    docs=documents, tensor_fields=["text_field"], index_name=self.unstructured_text_index
                )
            )

        correct_model_properties = get_model_properties("hf/all-MiniLM-L6-v2")
        self.index_management.update_index_settings_by_settings_dict(
            index_name=self.unstructured_text_index,
            settings_dict={"modelProperties": correct_model_properties}
        )

        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=documents, tensor_fields=["text_field"], index_name=self.unstructured_text_index
            )
        )

        updated_index = self.index_management.get_index(self.unstructured_text_index)
        self.assertEqual(correct_model_properties, updated_index.model.properties)
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.unstructured_text_index).number_of_documents)
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.unstructured_text_index).number_of_vectors)

    def test_updated_model_properties_for_structured_image_index(self):
        """
        Test that updating the model properties of an unstructured image index works correctly.
        """
        documents = [
            {
                "_id": "doc1",
                "image_field": TestImageUrls.IMAGE0.value,
                "text_field": "A sample text"
            }
        ]
        with self.assertRaises(InferenceError):
            self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    docs=documents, index_name=self.structured_image_index
                )
            )

        correct_model_properties = get_model_properties("open_clip/ViT-B-32/laion2b_s34b_b79k")
        self.index_management.update_index_settings_by_settings_dict(
            index_name=self.structured_image_index,
            settings_dict={"modelProperties": correct_model_properties}
        )

        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=documents, index_name=self.structured_image_index
            )
        )

        updated_index = self.index_management.get_index(self.structured_image_index)
        self.assertEqual(correct_model_properties, updated_index.model.properties)
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_image_index).number_of_documents)
        self.assertEqual(2, self.monitoring.get_index_stats_by_name(self.structured_text_index).number_of_vectors)
        
    def test_updated_model_properties_for_structured_text_index(self):
        """
        Test that updating the model properties of an unstructured image index works correctly.
        """
        documents = [
            {
                "_id": "doc1",
                "text_field": "A sample text"
            }
        ]
        with self.assertRaises(InferenceError):
            self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    docs=documents, index_name=self.structured_text_index
                )
            )

        correct_model_properties = get_model_properties("hf/all-MiniLM-L6-v2")
        self.index_management.update_index_settings_by_settings_dict(
            index_name=self.structured_text_index,
            settings_dict={"modelProperties": correct_model_properties}
        )

        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=documents, index_name=self.structured_text_index
            )
        )

        updated_index = self.index_management.get_index(self.structured_text_index)
        self.assertEqual(correct_model_properties, updated_index.model.properties)
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_text_index).number_of_documents)
        self.assertEqual(1, self.monitoring.get_index_stats_by_name(self.structured_text_index).number_of_vectors)