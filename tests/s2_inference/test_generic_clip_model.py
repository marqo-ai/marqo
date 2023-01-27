import numpy as np

from marqo.errors import IndexNotFoundError
from marqo.s2_inference.errors import InvalidModelPropertiesError, UnknownModelError, ModelLoadError
from marqo.tensor_search import tensor_search
from marqo.s2_inference.processing.custom_clip_utils import download_pretrained_from_url

from marqo.s2_inference.s2_inference import (
    available_models,
    vectorise,
    _validate_model_properties,
    _update_available_models
)

from tests.marqo_test import MarqoTestCase


class TestGenericModelSupport(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass


    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_2)
        except IndexNotFoundError as e:
            pass


    def test_create_index_with_custom_open_clip_model_properties_url(self):
        """index should get created with custom model_properties
        """
        model_name = 'test-model-1'
        model_properties = {"name": "open_clip custom model",
                            "dimensions": 512,
                            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
                            "type": "clip",
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


    def test_create_index_with_custom_openai_clip_model_properties_url(self):
        model_name = 'test-model-2'
        model_properties = {"name": "openai custom model",
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


    def test_create_index_with_custom_open_clip_model_properties_localpath(self):
        """index should get created with custom model_properties
        """
        url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
        target_model = download_pretrained_from_url(url)

        model_name = 'test-model-1'
        model_properties = {"name": "open_clip custom model",
                            "dimensions": 512,
                            "localpath": target_model,
                            "type": "clip",
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

    def test_vectorise_with_custom_open_clip_model_properties_invalid_localpath(self):
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

        self.assertRaises(ModelLoadError, vectorise, model_name, content, model_properties)


    def test_vectorise_with_custom_open_clip_model_properties_invalid_url(self):
        """index should get created with custom model_properties
        """
        content = ["testtest"]
        invalid_url = "http://test/test/test/testmodel.pt"

        model_name = 'test-model-1'
        model_properties = {"name": "open_clip custom model",
                            "dimensions": 512,
                            "url": invalid_url,
                            "type": "clip",
                            }

        self.assertRaises(ModelLoadError, vectorise, model_name, content, model_properties)


    def test_create_index_with_model_properties_without_model_name(self):
        """
            create_vector_index should throw an error
            if model_properties are given without model_name
        """
        model_properties = {"name": "openai custom model",
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
                            "name": "openai custom model",
                            "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
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

        tensor_search.add_documents(config=config, index_name=index_name, docs=docs, auto_refresh=auto_refresh)


    def test_load_custom_clip_without_url_or_localpath(self):
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

        self.assertRaises(ModelLoadError, vectorise, model_name,content, model_properties)

        model_properties["url"] = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

        vectorise(model_name, content, model_properties)


    def test_vectorise_without_clip_type(self):
        """_validate_model_properties should throw an exception if required keys are not given.
        """
        content = ["test test"]
        model_name = "test-model"
        model_properties = {
                           "name": "openai custom model",
                           "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            #"type": "clip",
                            }

        self.assertRaises(ModelLoadError, vectorise, model_name,content, model_properties)

        model_properties["type"] = "clip"
        vectorise(model_name, content, model_properties)


    def test_validate_model_properties_unknown_model_error(self):
        pass
        """_validate_model_properties should throw an error if model is not in registry,
            and if model_properties have not been given in index
        """
        model_name = "test-model"
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config,
            index_settings={
                "index_defaults": {
                    'model': model_name,
                    'type' : "clip"
                }
            }
        )

        model_properties = None

        self.assertRaises(UnknownModelError, _validate_model_properties, model_name, model_properties)


    def test_vectorise_custom_openai_clip_encode_image_results(self):

        epsilon = 1e-7

        image = "https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg"

        model_name = "test-model"
        model_properties = {
                            "name": "openai custom model",
                            "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            "type": "clip",
                            }

        a = vectorise(model_name, content = image, model_properties = model_properties)
        b = vectorise("ViT-B/32", content = image)

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon


    def test_vectorise_custom_openai_clip_encode_text_results(self):

        epsilon = 1e-7
        text = "this is a test to test the custom clip output results"

        model_name = "test-model"
        model_properties = {
                            "name": "openai custom model",
                            "dimensions": 512,
                            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                            "type": "clip",
                            }

        a = vectorise(model_name, content=text, model_properties=model_properties)
        b = vectorise("ViT-B/32", content=text)

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon

    def test_vectorise_custom_open_clip_encode_image_results(self):

        epsilon = 1e-7

        image = "https://raw.githubusercontent.com/marqo-ai/marqo-clip-onnx/main/examples/coco.jpg"

        model_name = "test-model"
        model_properties = {
                            "name": "open_clip custom model",
                            "dimensions": 512,
                            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
                            "type": "clip",
                            "jit" : False
                            }

        a = vectorise(model_name, content = image, model_properties = model_properties)
        b = vectorise("open_clip/ViT-B-32-quickgelu/laion400m_e31", content = image)

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon


    def test_vectorise_custom_open_clip_encode_text_results(self):
        epsilon = 1e-7
        text = "this is a test to test the custom clip output results"

        model_name = "test-model"
        model_properties = {
            "name": "ViT-B-32-quickgelu",
            "dimensions": 512,
            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
            "type": "clip",
            "jit": False
        }


        a = vectorise(model_name, content=text, model_properties=model_properties)
        b = vectorise("open_clip/ViT-B-32-quickgelu/laion400m_e31", content=text)

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon

    def test_unsupported_generic_clip_name(self):
        epsilon = 1e-7
        text = "this is a test to test the custom clip output results"

        model_name = "test-model"
        model_properties = {
            "name": "this is a test name",
            "dimensions": 512,
            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
            "type": "clip",
            "jit": False
        }

        a = vectorise(model_name, content=text, model_properties=model_properties)
        b = vectorise("open_clip/ViT-B-32-quickgelu/laion400m_e31", content=text)

        assert np.abs(np.array(a) - np.array(b)).sum() < epsilon


