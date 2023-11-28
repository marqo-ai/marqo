import math
import os
import sys
from tests.utils.transition import add_docs_caller, add_docs_batched
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from unittest import mock
from marqo.s2_inference.s2_inference import available_models, _create_model_cache_key, _validate_model_properties, clear_loaded_models, vectorise
import numpy as np
from marqo.tensor_search import utils
from marqo.tensor_search.models.private_models import S3Auth, ModelAuth, HfAuth
import typing
from marqo.tensor_search.models.search import Qidx, JHash, SearchContext, VectorisedJobs, VectorisedJobPointer
from marqo.tensor_search.enums import TensorField, SearchMethod, EnvVars, IndexSettingsField, MlModel
from marqo.errors import (
    MarqoApiError, MarqoError, IndexNotFoundError, InvalidArgError,
    InvalidFieldNameError, IllegalRequestedDocCount, BadRequestError, InternalError
)
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import copy
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.index_info import IndexInfo
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity
import unittest
import pprint
from marqo._httprequests import HttpRequests


def pass_through_vectorise(*args, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the call list
            via mock
            """
            return vectorise(*args, **kwargs)

class TestIndexWithSearchModel(MarqoTestCase):
    """
    Tests that add documents and search work as expected when using an index `search_model`,
    Even when it differs from the index `model`.
    """
    def setUp(self) -> None:
        self.index_with_search_model_registry = "my-index-with-search-model-registry"
        self.index_with_search_model_custom = "my-index-with-search-model-custom"
        self.index_with_model_auth = "my-index-with-model-auth"
        self.index_with_no_model = "my-index-with-no-model"
        self.index_with_default_settings = "my-index-with-default-settings"

        self._delete_test_indices()
        self._create_test_indices()

        # Sample backend response from OpenSearch (with no search_model for backwards compatibility test)
        self.sample_old_backend_response = {
            self.index_with_default_settings: {
                'mappings': {
                    '_meta': {
                        'index_settings': {
                            'index_defaults': {
                                'ann_parameters': {
                                    'engine': 'lucene',
                                    'name': 'hnsw',
                                    'parameters': {
                                        'ef_construction': 128,
                                        'm': 16
                                    },
                                    'space_type': 'cosinesimil'
                                },
                                'image_preprocessing': {'patch_method': None},
                                'model': 'ViT-L/14',
                                'normalize_embeddings': True,
                                'text_preprocessing': {'split_length': 2,
                                                        'split_method': 'sentence',
                                                        'split_overlap': 0},
                                'treat_urls_and_pointers_as_images': True
                            },
                            'number_of_replicas': 0,
                            'number_of_shards': 3
                        },
                        'media_type': 'text',
                        'model': 'ViT-L/14'
                    },
                    'dynamic_templates': [{'strings': {'mapping': {'type': 'text'}, 'match_mapping_type': 'string'}}],
                    'properties': {'Title': {'type': 'text'},
                    '__chunks': {
                        'properties': {
                            'Title': {
                                'ignore_above': 32766,
                                'type': 'keyword'
                            },
                            '__field_content': {'type': 'text'},
                            '__field_name': {'type': 'keyword'},
                            '__vector_marqo_knn_field': {
                                'dimension': 768,
                                'method': {
                                    'engine': 'lucene',
                                    'name': 'hnsw',
                                    'parameters': {
                                        'ef_construction': 128,
                                        'm': 16
                                    },
                                    'space_type': 'cosinesimil'
                                },
                                'type': 'knn_vector'
                            },
                            'captioned_image': {
                                'properties': {
                                    'caption': {
                                        'ignore_above': 32766,
                                        'type': 'keyword'
                                    },
                                    'image': {'ignore_above': 32766, 'type': 'keyword'}
                                }
                            }
                        },
                        'type': 'nested'
                    },
                    'captioned_image': {
                        'properties': {'caption': {'type': 'text'},
                                        'image': {'type': 'text'}
                                    }}}}}}

        # random search response
        self.sample_msearch_response = {
            'took': 12, 'responses': [
                {'took': 11, 'timed_out': False, '_shards': {'total': 3, 'successful': 3, 'skipped': 0, 'failed': 0}, 
                'hits': {
                    'total': {'value': 2, 'relation': 'eq'}, 
                    'max_score': 0.8582245, 
                    'hits': [
                        {
                            '_index': 'my-first-multimodal-index', 
                            '_id': '380f680e-628b-4d26-bfbf-73543db6e726', '_score': 0.8582245, 
                            '_source': {
                                'Title': 'Flying Plane', 
                                '__chunks': [{'__field_content': 'Flying Plane', '__field_name': 'Title', 'Title': 'Flying Plane', 'captioned_image': {'image': 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg', 'caption': 'An image of a passenger plane flying in front of the moon.'}}, 
                                             {'__field_content': '{"caption": "An image of a passenger plane flying in front of the moon.", "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"}', '__field_name': 'captioned_image', 'Title': 'Flying Plane', 'captioned_image': {'image': 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg', 'caption': 'An image of a passenger plane flying in front of the moon.'}}
                                            ], 
                                'captioned_image': {
                                    'image': 'https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg', 
                                    'caption': 'An image of a passenger plane flying in front of the moon.'
                                }
                            }, 
                            'inner_hits': {'__chunks': {'hits': {'total': {'value': 1, 'relation': 'eq'}, 'max_score': 0.8582245, 'hits': [{'_index': 'my-first-multimodal-index', '_id': '380f680e-628b-4d26-bfbf-73543db6e726', '_nested': {'field': '__chunks', 'offset': 0}, '_score': 0.8582245, '_source': {'__field_content': 'Flying Plane', '__field_name': 'Title'}}]}}}}]}, 'status': 200}]}
        # Any tests that call add_document, search, bulk_search need this env var
        # Ensure other os.environ patches in indiv tests do not erase this one.
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self):
        clear_loaded_models()
        self.device_patcher.stop()

    def _delete_test_indices(self, indices=None):
        if indices is None or not indices:
            ix_to_delete = [self.index_with_search_model_registry, self.index_with_search_model_custom, self.index_with_model_auth, self.index_with_no_model, self.index_with_default_settings]
        else:
            ix_to_delete = indices
        for ix_name in ix_to_delete:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix_name)
            except IndexNotFoundError as s:
                pass

    def _create_test_indices(self, indices=None):
        # Search model in registry
        tensor_search.create_vector_index(
            config=self.config, 
            index_name=self.index_with_search_model_registry,
            index_settings={
                IndexSettingsField.index_defaults: {
                    "model": "ViT-B/32",          # dimension is 512
                    "search_model": "onnx32/open_clip/ViT-B-32/laion2b_e16",
                }
            }
        )

        # Search model is custom
        tensor_search.create_vector_index(
            config=self.config, 
            index_name=self.index_with_search_model_custom,
            index_settings={
                IndexSettingsField.index_defaults: {
                    "model": "ViT-B/32",          # dimension is 512
                    "search_model": "my_custom_search_model",
                    "search_model_properties": {
                        "name": "ViT-B-32-quickgelu",
                        "dimensions": 512,
                        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                        "type": "open_clip",
                    }
                }
            }
        )

    def test_search_model_loading_from_registry_with_eject(self):
        """
        Ensures that search model is loaded correctly when using add_documents then search.
        Ejected models in between to isolate model list.
        """
        # Registry models
        # Add documents
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_with_search_model_registry,
                docs=[
                    {"_id": "correct_doc", "description": "lemon"},
                    {"_id": "dummy_doc_1", "description": "DUMMY RESULT"},
                    {"_id": "dummy_doc_2", "description": "DONT SEARCH FOR ME LOL"},
                ],
                auto_refresh=True, device="cpu", 
            )
        )

        # Assert only ViT-B/32 is loaded (this is `model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 1
        assert loaded_models[0]["model_name"] == "ViT-B/32"     # onnx32/open_clip/ViT-B-32/laion2b_e16 should not be loaded
        assert loaded_models[0]["model_device"] == "cpu"

        # Eject all models
        tensor_search.eject_model(model_name="ViT-B/32", device="cpu")

        # Search
        search_result = tensor_search.search(
            config=self.config, index_name=self.index_with_search_model_registry,
            text="a yellow fruit", device="cpu"
        )

        # Assert now `onnx32/open_clip/ViT-B-32/laion2b_e16` is also loaded (this is `search_model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 1
        assert loaded_models[0]["model_name"] == "onnx32/open_clip/ViT-B-32/laion2b_e16"     # ViT-B-32 should not be loaded
        assert loaded_models[0]["model_device"] == "cpu"

        # Assert correct result found
        assert search_result["hits"][0]["_id"] == "correct_doc"
    
    def test_search_model_loading_from_registry(self):
        """
        Ensures that search model is loaded correctly when using search then add_documents
        """
        # Search first
        search_result = tensor_search.search(
            config=self.config, index_name=self.index_with_search_model_registry,
            text="a yellow fruit", device="cpu"
        )

        # Assert `onnx32/open_clip/ViT-B-32/laion2b_e16` is loaded (this is `search_model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 1
        assert loaded_models[0]["model_name"] == "onnx32/open_clip/ViT-B-32/laion2b_e16"     # ViT-B-32 should not be loaded
        assert loaded_models[0]["model_device"] == "cpu"

        # Add documents
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_with_search_model_registry,
                docs=[
                    {"_id": "correct_doc", "description": "lemon"},
                    {"_id": "dummy_doc_1", "description": "DUMMY RESULT"},
                    {"_id": "dummy_doc_2", "description": "DONT SEARCH FOR ME LOL"},
                ],
                auto_refresh=True, device="cpu", 
            )
        )

        # Assert ViT-B/32 is also loaded (this is `model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 2
        assert loaded_models[1]["model_name"] == "ViT-B/32"     # onnx32/open_clip/ViT-B-32/laion2b_e16 should not be loaded
        assert loaded_models[1]["model_device"] == "cpu"
        
    def test_search_model_loading_custom_with_eject(self):
        """
        Ensures that custom search model is loaded correctly when using add_documents then search.
        Ejects add docs model to isolate model list for search.
        """
        # Registry models
        # Add documents
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_with_search_model_custom,
                docs=[
                    {"_id": "correct_doc", "description": "lemon"},
                    {"_id": "dummy_doc_1", "description": "DUMMY RESULT"},
                    {"_id": "dummy_doc_2", "description": "DONT SEARCH FOR ME LOL"},
                ],
                auto_refresh=True, device="cpu", 
            )
        )

        # Assert only ViT-B/32 is loaded (this is `model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 1
        assert loaded_models[0]["model_name"] == "ViT-B/32"     # my_custom_search_model should not be loaded
        assert loaded_models[0]["model_device"] == "cpu"

        # Eject all models
        tensor_search.eject_model(model_name="ViT-B/32", device="cpu")

        # Search
        search_result = tensor_search.search(
            config=self.config, index_name=self.index_with_search_model_custom,
            text="a yellow fruit", device="cpu"
        )

        # Assert now `my_custom_search_model` is also loaded (this is `search_model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 1
        assert loaded_models[0]["model_name"] == "my_custom_search_model"     # ViT-B-32 should not be loaded
        assert loaded_models[0]["model_device"] == "cpu"
    
    def test_search_model_loading_custom(self):
        """
        Ensures that custom search model is loaded correctly when using search first then add_documents
        """
        # Registry models

        # Search first
        search_result = tensor_search.search(
            config=self.config, index_name=self.index_with_search_model_custom,
            text="a yellow fruit", device="cpu"
        )

        # Assert `my_custom_search_model` is loaded (this is `search_model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 1
        assert loaded_models[0]["model_name"] == "my_custom_search_model"     # ViT-B-32 should not be loaded
        assert loaded_models[0]["model_device"] == "cpu"

        # Add documents
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_with_search_model_custom,
                docs=[
                    {"_id": "correct_doc", "description": "lemon"},
                    {"_id": "dummy_doc_1", "description": "DUMMY RESULT"},
                    {"_id": "dummy_doc_2", "description": "DONT SEARCH FOR ME LOL"},
                ],
                auto_refresh=True, device="cpu", 
            )
        )

        # Assert both models are loaded (this is `model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 2
        assert loaded_models[1]["model_name"] == "ViT-B/32"     # my_custom_search_model should not be loaded
        assert loaded_models[1]["model_device"] == "cpu"
    
    def test_search_model_used_for_vectorise(self):
        """
        Ensures that searching uses the correct model in vectorisation call
        """

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            return tensor_search.search(
                config=self.config, index_name=self.index_with_search_model_custom, text="random query",
            )
        
        run()
        assert len(mock_vectorise.call_args_list) == 1
        args, kwargs = mock_vectorise.call_args_list[0]
        assert kwargs["model_name"] == "my_custom_search_model"
        assert kwargs["model_properties"] == {
            "name": "ViT-B-32-quickgelu",
            "dimensions": 512,
            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
            "type": "open_clip",
        }


    def test_search_model_bulk_search(self):
        """
        Ensures that custom search model is loaded correctly when using bulk_search
        Add docs first, then bulk search
        """
        # Add documents
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_with_search_model_custom,
                docs=[
                    {"_id": "correct_doc", "description": "lemon"},
                    {"_id": "dummy_doc_1", "description": "DUMMY RESULT"},
                    {"_id": "dummy_doc_2", "description": "DONT SEARCH FOR ME LOL"},
                ],
                auto_refresh=True, device="cpu", 
            )
        )

        # Assert only ViT-B/32 is loaded (this is `model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 1
        assert loaded_models[0]["model_name"] == "ViT-B/32"     # my_custom_search_model should not be loaded
        assert loaded_models[0]["model_device"] == "cpu"

        # Bulk search
        search_result = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[
                BulkSearchQueryEntity(index=self.index_with_search_model_custom, q="a yellow fruit")
            ]),
            marqo_config=self.config,
        )

        # Assert now `my_custom_search_model` is also loaded (this is `search_model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 2
        assert loaded_models[1]["model_name"] == "my_custom_search_model"     # ViT-B-32 should not be loaded
        assert loaded_models[1]["model_device"] == "cpu"

        assert search_result["result"][0]["hits"][0]["_id"] == "correct_doc"
    
    def test_search_model_auth_hf_search(self):
        """
        Ensures that model auth works properly when searching with a custom search_model
        """
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        search_model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        hf_settings = {
            "index_defaults": {
                "model": "ViT-B/32",
                "search_model": "my_custom_search_model",
                "search_model_properties": search_model_properties,
            }
        }

        tensor_search.create_vector_index(config=self.config, index_name=self.index_with_model_auth, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.pt'

        mock_open_clip_create_model = mock.MagicMock()

        with unittest.mock.patch('open_clip.create_model_and_transforms', mock_open_clip_create_model):
            with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                try:
                    res = tensor_search.search(
                        config=self.config, text='hello', index_name=self.index_with_model_auth,
                        model_auth=ModelAuth(hf=HfAuth(token=hf_token)))
                except BadRequestError:
                    # bad request due to no models actually being loaded
                    pass

        mock_hf_hub_download.assert_called_once_with(
            token=hf_token,
            repo_id=hf_repo_name,
            filename=hf_object,
            cache_dir = None,
        )

        # is the open clip model being loaded with the expected args?
        called_with_expected_args = any(
            call.kwargs.get("pretrained") == "cache/path/to/model.pt"
            and call.kwargs.get("model_name") == "ViT-B/32"
            for call in mock_open_clip_create_model.call_args_list
        )
        assert len(mock_open_clip_create_model.call_args_list) == 1
        assert called_with_expected_args, "Expected call not found"

    def test_search_model_no_model(self):
        """
        Ensures that model can be no_model while search_model still does vectorisation
        """
        tensor_search.create_vector_index(
            config=self.config, 
            index_name=self.index_with_no_model,
            index_settings={
                IndexSettingsField.index_defaults: {
                    "model": "no_model",          # dimension is 512
                    "model_properties": {
                        "dimensions": 512
                    },
                    "search_model": "ViT-B/32",
                }
            }
        )

        search_result = tensor_search.search(
            config=self.config, index_name=self.index_with_no_model,
            text="a random query", device="cpu"
        )

        # Assert `my_custom_search_model` is loaded (this is `search_model`)
        loaded_models = tensor_search.get_loaded_models().get("models")
        assert len(loaded_models) == 1
        assert loaded_models[0]["model_name"] == "ViT-B/32"     # ViT-B-32 should not be loaded
        assert loaded_models[0]["model_device"] == "cpu"
    
    def test_backwards_compatible_search(self):
        """
        Ensure that when searching and `search_model` is NOT in backend response (meaning old index),
        search continues as normal using `model`.
        """

        # Create index with no settings
        tensor_search.create_vector_index(
            config=self.config, 
            index_name=self.index_with_default_settings,
            index_settings={}
        )
        
        def replace_backend_mappings_response(*args, **kwargs):
            """
            Replace the mappings response with a sample old response from OpenSearch (no search_model)
            Otherwise, behave as usual
            """
            if kwargs["path"].endswith("/_mapping"):
                return self.sample_old_backend_response     # this response has model = "ViT-L/14"
            elif kwargs["path"] == "_msearch":
                return self.sample_msearch_response

        mock__get = mock.MagicMock()
        mock__get.side_effect = replace_backend_mappings_response
        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
        def run():
            return tensor_search.search(
                config=self.config, index_name=self.index_with_default_settings, text="random query",
            )
        
        run()

        # Must be called with `model`, which is ViT-L/14
        assert len(mock_vectorise.call_args_list) == 1
        args, kwargs = mock_vectorise.call_args_list[0]
        assert kwargs["model_name"] == "ViT-L/14"
        assert kwargs["model_properties"] == {
            "name": "ViT-L/14",
            "dimensions": 768,
            "notes": "CLIP ViT-L/14",
            "type":"clip"
        }
    

class TestSearchModelUtils(MarqoTestCase):
    def test_determine_model_for_search_vectorisation(self):
        """
        Should use search model when available.
        Use model otherwise.
        """
        info_with_search_model = IndexInfo(
            model_name="model_that_exists", 
            search_model_name="search_model_that_exists", properties={},
            index_settings={
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False, 
                    "normalize_embeddings": False,
                    'model': "model_that_exists",
                    'model_properties': {"dimensions": 12345},
                    'search_model': "search_model_that_exists",
                    'search_model_properties': {"dimensions": 6789},
                }
            }
        )

        info_with_no_search_model = IndexInfo(
            model_name="model_that_exists", 
            search_model_name=None, properties={},
            index_settings={
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False, 
                    "normalize_embeddings": False,
                    'model': "model_that_exists",
                    'model_properties': {"dimensions": 12345},
                },
            }
        )

        assert tensor_search.determine_model_for_search_vectorisation(info_with_search_model) == \
            ("search_model_that_exists", {"dimensions": 6789})
        
        assert tensor_search.determine_model_for_search_vectorisation(info_with_no_search_model) == \
            ("model_that_exists", {"dimensions": 12345})