import os
import unittest
from typing import Dict

from marqo.core.exceptions import IndexNotFoundError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search.api import create_index
from marqo.tensor_search.models.index_settings import IndexSettings
from integ_tests.marqo_test import MarqoTestCase, TestImageUrls



class TestPrivateModelLoading(MarqoTestCase):
    """A test class for loading private models end to end in Marqo."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.aws_access_key_id = os.getenv("PRIVATE_MODEL_TESTS_AWS_ACCESS_KEY_ID", None)
        cls.aws_secret_access_key = os.getenv("PRIVATE_MODEL_TESTS_AWS_SECRET_ACCESS_KEY", None)
        cls.hf_token = os.getenv("PRIVATE_MODEL_TESTS_HF_TOKEN", None)

        if any([cls.aws_access_key_id is None, cls.aws_secret_access_key is None, cls.hf_token is None]):
            raise ValueError("Please set the AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
                             "and HF_TOKEN environment variables to run this test.")

        cls.index_name = "test_index_private_model_loading"

    def setUp(self):
        super().setUp()
        try:
            self.index_management.delete_index_by_name(self.index_name)
        except IndexNotFoundError:
            pass

    def tearDown(self):
        super().setUp()
        try:
            self.index_management.delete_index_by_name(self.index_name)
        except IndexNotFoundError:
            pass

    def _help_test_index(self, model: str, model_properties: Dict):
        index_settings = IndexSettings(
            model=model,
            modelProperties=model_properties,
            type="unstructured",
            treatUrlsAndPointersAsMedia=True
        )
        create_index(self.index_name, index_settings, self.config)

    def test_load_private_hf_model_from_a_private_zip_file_on_s3(self):
        model = "private-e5-zip-on-s3"
        model_properties = {
            "dimensions": 768,
            "type": "hf",
            "modelLocation": {
                "s3": {
                    "Bucket": "marqo-opensource-private-model-tests",
                    "Key": "private-e5-model.zip"
                },
                "auth_required": True
            }
        }
        self._help_test_index(model, model_properties)
        add_docs_params = AddDocsParams(
            index_name=self.index_name,
            docs=[{
                "id": "1",
                "text": "This is a test document."
            }],
            tensor_fields = ["text"],
            model_auth={
                "s3": {
                    "aws_access_key_id": self.aws_access_key_id,
                    "aws_secret_access_key": self.aws_secret_access_key
                }
            }
        )

        res = self.add_documents(self.config, add_docs_params= add_docs_params)
        self.assertEqual(False, res.errors)
        self.assertEqual(self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents, 1)

    def test_load_private_hf_model_from_a_private_hf_repo(self):
        model = "private-e5-repo-on-hf"
        model_properties = {
            "dimensions": 768,
            "type": "hf",
            "modelLocation": {
                "hf": {
                    "repoId": "Marqo/e5-base-v2-private-test"
                },
                "auth_required": True
            }
        }
        self._help_test_index(model, model_properties)
        add_docs_params = AddDocsParams(
            index_name=self.index_name,
            docs=[{
                "id": "1",
                "text": "This is a test document."
            }],
            tensor_fields = ["text"],
            model_auth={
                "hf": {"token": self.hf_token}
            }
        )

        res = self.add_documents(self.config, add_docs_params= add_docs_params)
        self.assertEqual(False, res.errors)
        self.assertEqual(self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents, 1)

    def test_load_private_open_clip_model_from_a_private_ckpt_on_s3(self):
        model = "private-marqo-fashion-clip-model-ckpt-on-s3"
        model_properties = {
            "dimensions": 512,
            "name": "ViT-B-16",
            "type": "open_clip",
            "modelLocation": {
                "s3": {
                    "Bucket": "marqo-opensource-private-model-tests",
                    "Key": "private-fashion-clip-ckpt.bin"
                },
                "auth_required": True
            }
        }
        self._help_test_index(model, model_properties)
        add_docs_params = AddDocsParams(
            index_name=self.index_name,
            docs=[{
                "id": "1",
                "text": "This is a test document.",
                "image": str(TestImageUrls.IMAGE2)
            }],
            tensor_fields=["text", "image"],
            model_auth={
                "s3": {
                    "aws_access_key_id": self.aws_access_key_id,
                    "aws_secret_access_key": self.aws_secret_access_key
                }
            }
        )
        res = self.add_documents(self.config, add_docs_params=add_docs_params)
        self.assertEqual(False, res.errors)
        self.assertEqual(self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents, 1)
        self.assertEqual(self.monitoring.get_index_stats_by_name(self.index_name).number_of_vectors, 2)

    def test_load_private_open_clip_model_from_a_private_ckpt_on_hf(self):
        model = "private-marqo-fashion-siglip-model-ckpt-on-hf"
        model_properties = {
            "dimensions": 768,
            "name": "ViT-B-16-SigLIP",
            "type": "open_clip",
            "modelLocation": {
                "hf": {
                    "repoId": "Marqo/private-ecommerce-embeddings-B",
                    "filename": "open_clip_pytorch_model.bin"
                },
                "auth_required": True
            }
        }
        self._help_test_index(model, model_properties)
        add_docs_params = AddDocsParams(
            index_name=self.index_name,
            docs=[{
                "id": "1",
                "text": "This is a test document.",
                "image": str(TestImageUrls.IMAGE2)
            }],
            tensor_fields=["text", "image"],
            model_auth={
                "hf": {"token": self.hf_token}
            }
        )
        res = self.add_documents(self.config, add_docs_params=add_docs_params)
        self.assertEqual(False, res.errors)
        self.assertEqual(self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents, 1)
        self.assertEqual(self.monitoring.get_index_stats_by_name(self.index_name).number_of_vectors, 2)