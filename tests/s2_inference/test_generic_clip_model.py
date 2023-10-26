import numpy as np
import os
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.errors import IndexNotFoundError
from marqo.s2_inference.errors import UnknownModelError, ModelLoadError
from marqo.tensor_search import tensor_search
from marqo.s2_inference.processing.custom_clip_utils import download_pretrained_from_url
from marqo.s2_inference.s2_inference import clear_loaded_models
from marqo.s2_inference.s2_inference import (
    vectorise,
    _validate_model_properties
)

from tests.marqo_test import MarqoTestCase
from unittest import mock


class TestGenericModelSupport(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass
        
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_2)
        except IndexNotFoundError as e:
            pass
        clear_loaded_models()
        self.device_patcher.stop()


    def test_create_index_and_add_documents_with_generic_open_clip_model_properties_url(self):
        """index should get created with custom model_properties
        """
        # Step1 - Create Index
        score_threshold = 0.6
        model_name = 'test-model-1'
        model_properties = {"name": "ViT-B-32-quickgelu",
                            "dimensions": 512,
                            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                            "type": "open_clip",
                            }
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': model_name,
                    'model_properties': model_properties
                }
            }
        )
        # Step2 - Add documents
        docs = [
            {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }]

        auto_refresh = True
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=docs, auto_refresh=auto_refresh, device="cpu")
        )

        # test if we can get the document by _id
        assert tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123") == {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }

        # test another document
        docs2 = [
            {
                "_id": "321",
                "title 1": "test test test",
                "desc 2": "test again test again test again"
            }]

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=docs2, auto_refresh=auto_refresh, device="cpu"))

        assert tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="321") == {
                "_id": "321",
                "title 1": "test test test",
                "desc 2": "test again test again test again"
               }


        # Step3 - Search
        search_res = tensor_search.search(config=self.config, index_name=self.index_name_1, text = "content 2. blah blah blah", result_count=1)
        assert len(search_res['hits']) == 1
        assert search_res["hits"][0]["_score"] > score_threshold
        assert search_res["hits"][0]["_id"] == "123"


    def test_pipeline_with_generic_openai_clip_model_properties_url(self):
        score_threshold = 0.6
        model_name = 'test-model-2'
        model_properties = {"name": "ViT-B/32",
                            "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            "type": "clip",
                            }

        tensor_search.create_vector_index(
            index_name=self.index_name_2, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': model_name,
                    'model_properties': model_properties
                }
            }
        )

        docs = [
            {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }]

        auto_refresh = True
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_2, docs=docs, auto_refresh=auto_refresh, device="cpu"
        ))

        assert tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_2,
            document_id="123") == {
                   "_id": "123",
                   "title 1": "content 1",
                   "desc 2": "content 2. blah blah blah"
               }

        docs2 = [
            {
                "_id": "321",
                "title 1": "test test test",
                "desc 2": "test again test again test again"
            }]

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_2, docs=docs2, auto_refresh=auto_refresh, device="cpu"))

        assert tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_2,
            document_id="321") == {
                "_id": "321",
                "title 1": "test test test",
                "desc 2": "test again test again test again"
               }

        search_res = tensor_search.search(config=self.config, index_name=self.index_name_2,
                                          text="content 2. blah blah blah", result_count=1)
        assert len(search_res['hits']) == 1
        assert search_res["hits"][0]["_score"] > score_threshold
        assert search_res["hits"][0]["_id"] == "123"


    def test_pipeline_with_generic_open_clip_model_properties_localpath(self):
        """index should get created with custom model_properties
        """
        score_threshold = 0.6
        url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
        target_model = download_pretrained_from_url(url)

        model_name = 'test-model-1'
        model_properties = {"name": "ViT-B-32-quickgelu",
                            "dimensions": 512,
                            "localpath": target_model,
                            "type": "open_clip",
                            }
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': model_name,
                    'model_properties': model_properties
                }
            }
        )

        docs = [
            {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }]

        auto_refresh = True
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=docs, auto_refresh=auto_refresh, device="cpu"))

        assert tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123") == {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }

        docs2 = [
            {
                "_id": "321",
                "title 1": "test test test",
                "desc 2": "test again test again test again"
            }]

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=docs2, auto_refresh=auto_refresh, device="cpu"))

        assert tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="321") == {
                "_id": "321",
                "title 1": "test test test",
                "desc 2": "test again test again test again"
               }

        # Step3 - Search
        search_res = tensor_search.search(config=self.config, index_name=self.index_name_1, text = "content 2. blah blah blah", result_count=1)
        assert len(search_res['hits']) == 1
        assert search_res["hits"][0]["_score"] > score_threshold
        assert search_res["hits"][0]["_id"] == "123"


    def test_vectorise_with_generic_open_clip_model_properties_invalid_localpath(self):
        """index should get created with custom model_properties
        """
        content = ["testtest"]
        invalid_localpath = "/test/test/test/testmodel.pt"

        model_name = 'test-model-1'
        model_properties = {"name": "open_clip custom model",
                            "dimensions": 512,
                            "localpath": invalid_localpath,
                            "type": "clip",
                            }

        self.assertRaises(ModelLoadError, vectorise, model_name, content, model_properties, device="cpu")


    def test_vectorise_with_generic_open_clip_model_properties_invalid_url(self):
        """index should get created with custom model_properties
        """
        content = ["testtest"]
        invalid_url = "http://test/test/test/testmodel.pt"

        model_name = 'test-model-1'
        model_properties = {"name": "test test void model",
                            "dimensions": 512,
                            "url": invalid_url,
                            "type": "clip",
                            }

        self.assertRaises(ModelLoadError, vectorise, model_name, content, model_properties, device="cpu")


    def test_create_index_with_model_properties_without_model_name(self):
        """
            create_vector_index should throw an error
            if model_properties are given without model_name
        """
        model_properties = {"name": "ViT-B-32",
                            "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            "type": "clip",
                            }

        index_settings = {
            "index_defaults": {
                # 'model': model_name,
                'model_properties': model_properties
            }
        }

        self.assertRaises(UnknownModelError, tensor_search.create_vector_index, config=self.config,
                         index_name=self.index_name_1, index_settings=index_settings)


    def test_add_documents_text_and_image(self):
        """if given the right input, add_documents should work without any throwing any errors
        """
        model_name = "test-model"
        model_properties = {
                            "name": "ViT-B/16",
                            "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
                            "type": "clip",
                            }
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': model_name,
                    'model_properties': model_properties,
                    "treat_urls_and_pointers_as_images": True
                }
            }
        )

        config = self.config
        index_name = self.index_name_1
        docs = [
            {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah",
                "image" : "https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg"
            }]
        auto_refresh = True

        tensor_search.add_documents(config=config, add_docs_params=AddDocsParams(
            index_name=index_name, docs=docs, auto_refresh=auto_refresh, device="cpu"))


    def test_load_generic_clip_without_url_or_localpath(self):
        """vectorise should throw an exception if url or localpath are not given.
        """
        content = ["test test"]
        model_name = "test-model"
        model_properties = {
                           "name": "openai custom model",
                           "dimensions": 512,
                            #"url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            "type": "clip",
                            }

        self.assertRaises(ModelLoadError, vectorise, model_name,content, model_properties, device="cpu")

        model_properties["url"] = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

        vectorise(model_name, content, model_properties, device="cpu")


    def test_vectorise_without_clip_type(self):
        """_validate_model_properties should throw an exception if required keys are not given.
        """
        content = ["test test"]
        model_name = "test-model"
        model_properties = {
                           "name": "ViT-B-32",
                           "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            #"type": "clip",
                            }

        self.assertRaises(ModelLoadError, vectorise, model_name,content, model_properties, device="cpu")

        model_properties["type"] = "clip"
        vectorise(model_name, content, model_properties, device="cpu")


    def test_vectorise_generic_openai_clip_encode_image_results(self):

        epsilon = 1e-7

        image = "https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg"

        model_name = "test-model"
        model_properties = {
                            "name": "ViT-B/32",
                            "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            "type": "clip",
                            }

        a = vectorise(model_name, content = image, model_properties = model_properties, device="cpu")
        b = vectorise("ViT-B/32", content = image, device="cpu")

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon


    def test_vectorise_generic_openai_clip_encode_text_results(self):

        epsilon = 1e-7
        text = "this is a test to test the custom clip output results"

        model_name = "test-model"
        model_properties = {
                            "name": "ViT-B/32",
                            "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            "type": "clip",
                            }

        a = vectorise(model_name, content=text, model_properties=model_properties, device="cpu")
        b = vectorise("ViT-B/32", content=text, device="cpu")

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon


    def test_vectorise_generic_open_clip_encode_image_results(self):

        epsilon = 1e-7

        image = "https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg"

        model_name = "test-model"
        model_properties = {
                            "name": "ViT-B-32-quickgelu",
                            "dimensions": 512,
                            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
                            "type": "open_clip",
                            "jit" : False
                            }

        a = vectorise(model_name, content = image, model_properties = model_properties, device="cpu")
        b = vectorise("open_clip/ViT-B-32-quickgelu/laion400m_e31", content = image, device="cpu")

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon


    def test_vectorise_generic_open_clip_encode_text_results(self):
        epsilon = 1e-7
        text = "this is a test to test the custom clip output results"

        model_name = "test-model"
        model_properties = {
            "name": "ViT-B-32-quickgelu",
            "dimensions": 512,
            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
            "type": "open_clip",
            "jit": False
        }


        a = vectorise(model_name, content=text, model_properties=model_properties, device="cpu")
        b = vectorise("open_clip/ViT-B-32-quickgelu/laion400m_e31", content=text, device="cpu")

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon


    def test_incorrect_vectorise_generic_open_clip_encode_text_results(self):
        epsilon = 1e-3
        text = "this is a test to test the custom clip output results"

        model_name = "test-model"
        model_properties = {
            "name": "ViT-B-32-quickgelu",
            "dimensions": 512,
            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
            "type": "open_clip",
            "jit": False
        }


        a = vectorise(model_name, content=text, model_properties=model_properties, device="cpu")
        b = vectorise("open_clip/ViT-B-32-quickgelu/laion400m_e32", content=text, device="cpu")

        assert np.abs(np.array(a) - np.array(b)).sum() > epsilon



