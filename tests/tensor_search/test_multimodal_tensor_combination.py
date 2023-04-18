import unittest.mock
import pprint

import torch

import marqo.tensor_search.backend
from marqo.errors import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.tensor_search import add_documents, vectorise_multimodal_combination_field
from marqo.errors import DocumentNotFoundError
import numpy as np
from marqo.tensor_search.validation import validate_dict
from marqo.s2_inference.s2_inference import vectorise
import requests
from marqo.s2_inference.clip_utils import load_image_from_path
import json
from unittest.mock import patch
from marqo.errors import MarqoWebError


class TestMultimodalTensorCombination(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.mappings = {"combo_text_image" :{"type": "multimodal_combination", "weights" : {
            "text" : 0.5, "image" : 0.8}
        }}
        self.endpoint = self.authorized_url
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except:
            pass
    def test_add_documents(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        expected_doc = dict({
            "Title": "Horse rider",
            "combo_text_image": {
                "text": "A rider is riding a horse jumping over the barrier.",
                "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg"
            },
            "_id": "0"
        })
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            expected_doc,

            # this is just a dummy one
            {
                "Title": "Horse rider",
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "_id": "1"
            },
        ], mappings = {"combo_text_image" :{"type": "multimodal_combination", "weights" : {
            "text" : 0.5, "image" : 0.8}
        }},auto_refresh=True)

        added_doc = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="0",
                                                     show_vectors=True)
        for key, value in expected_doc.items():
            assert expected_doc[key] == added_doc[key]

        tensor_field = added_doc["_tensor_facets"]

        # for "Title" : "Horse Rider"
        assert "_embedding" in tensor_field[0]
        assert tensor_field[0]["Title"] == expected_doc["Title"]
        #
        # for combo filed
        assert "_embedding" in tensor_field[1]
        assert tensor_field[1]["combo_text_image"] == json.dumps(expected_doc["combo_text_image"])

    def test_multimodal_tensor_combination_score(self):
        def get_score(document):
            try:
                tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            except IndexNotFoundError as e:
                pass

            tensor_search.create_vector_index(
                index_name=self.index_name_1, config=self.config, index_settings={
                    IndexSettingsField.index_defaults: {
                        IndexSettingsField.model: "ViT-B/32",
                        IndexSettingsField.treat_urls_and_pointers_as_images: True,
                        IndexSettingsField.normalize_embeddings: False
                    }
                })

            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[document],
                                        auto_refresh=True, mappings = {"combo_text_image" : {"type":"multimodal_combination",
            "weights":{"image_field":0.5,"text_field":0.5}}})
            self.assertEqual(1, tensor_search.get_stats(config=self.config, index_name=self.index_name_1)[
                "numberOfDocuments"])
            res = tensor_search.search(config=self.config, index_name=self.index_name_1,
                                       text="", result_count=1, )

            return res["hits"][0]["_score"]

        score_1 = get_score({
            "text_field": "A rider is riding a horse jumping over the barrier.",
            # "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
        })

        score_2 = get_score({
            # "text_field": "A rider is riding a horse jumping over the barrier.",
            "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
        })

        score_3 = get_score({
            "combo_text_image": {
                "text_field" : "A rider is riding a horse jumping over the barrier.",
                "image_field" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
        },
        })

        assert (score_3 >= min(score_1, score_2)) and (score_3 <= max(score_1, score_2))

    def test_multimodal_tensor_combination_tensor_value(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "text_field_1": "A rider is riding a horse jumping over the barrier.",
                    "text_field_2": "What is the best to wear on the moon?",
                    "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                },
                "_id":"c1"
            },

            {
                "combo_text_image": {
                    "text_field_1": "A rider is riding a horse jumping over the barrier.",
                    "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    "text_field_2": "What is the best to wear on the moon?",
                    "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                },
                "_id": "c2"
            },

            {
                "combo_text_image": {
                    "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                    "text_field_1": "A rider is riding a horse jumping over the barrier.",
                    "text_field_2": "What is the best to wear on the moon?",
                },
                "_id": "c3"
            },

            {
                "combo_text_image": {
                    "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    "text_field_1": "A rider is riding a horse jumping over the barrier.",
                    "text_field_2": "What is the best to wear on the moon?",
                    "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                },
                "_id": "c4"
            },

            {
                "text_field_1": "A rider is riding a horse jumping over the barrier.",
                "_id": "1"
            },
            {
                "text_field_2": "What is the best to wear on the moon?",
                "_id": "2"
            },
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "_id": "3"
            },
            {
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                "_id": "4"
            },

        ], auto_refresh=True, mappings = {"combo_text_image" : {"type":"multimodal_combination",
            "weights":{"text_field_1": 0.32,"text_field_2": 0, "image_field_1" : -0.48, "image_field_2": 1.34}}})

        combo_tensor_1 = np.array(tensor_search.get_document_by_id(config=self.config,
                                                                 index_name=self.index_name_1, document_id="c1",
                                                                 show_vectors=True)['_tensor_facets'][0]["_embedding"])

        combo_tensor_2 = np.array(tensor_search.get_document_by_id(config=self.config,
                                                                 index_name=self.index_name_1, document_id="c2",
                                                                 show_vectors=True)['_tensor_facets'][0]["_embedding"])

        combo_tensor_3 = np.array(tensor_search.get_document_by_id(config=self.config,
                                                                 index_name=self.index_name_1, document_id="c3",
                                                                 show_vectors=True)['_tensor_facets'][0]["_embedding"])

        combo_tensor_4 = np.array(tensor_search.get_document_by_id(config=self.config,
                                                                 index_name=self.index_name_1, document_id="c4",
                                                                 show_vectors=True)['_tensor_facets'][0]["_embedding"])
        text_tensor_1 = \
            np.array(tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="1",
                                                      show_vectors=True)['_tensor_facets'][0]["_embedding"])
        text_tensor_2 = \
            np.array(tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="2",
                                                      show_vectors=True)['_tensor_facets'][0]["_embedding"])
        image_tensor_1 = \
            np.array(tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="3",
                                                      show_vectors=True)['_tensor_facets'][0]["_embedding"])
        image_tensor_2 = \
            np.array(tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="4",
                                                      show_vectors=True)['_tensor_facets'][0]["_embedding"])

        expected_tensor = np.mean([text_tensor_1 * 0.32, text_tensor_2 * 0, image_tensor_1 * -0.48, image_tensor_2 * 1.34], axis = 0)
        assert np.allclose(combo_tensor_1, expected_tensor, atol=1e-5)
        assert np.allclose(combo_tensor_2, expected_tensor, atol=1e-5)
        assert np.allclose(combo_tensor_3, expected_tensor, atol=1e-5)
        assert np.allclose(combo_tensor_4, expected_tensor, atol=1e-5)

    def test_multimodal_tensor_combination_zero_weight(self):
        def get_score(document):
            try:
                tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            except IndexNotFoundError as e:
                pass

            tensor_search.create_vector_index(
                index_name=self.index_name_1, config=self.config, index_settings={
                    IndexSettingsField.index_defaults: {
                        IndexSettingsField.model: "ViT-B/32",
                        IndexSettingsField.treat_urls_and_pointers_as_images: True,
                        IndexSettingsField.normalize_embeddings: False
                    }
                })

            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[document],
                                        auto_refresh=True, mappings = {"combo_text_image" : {"type":"multimodal_combination",
            "weights":{"image_field": 0,"text_field": 1}}})

            self.assertEqual(1, tensor_search.get_stats(config=self.config, index_name=self.index_name_1)[
                "numberOfDocuments"])
            res = tensor_search.search(config=self.config, index_name=self.index_name_1,
                                       text="", result_count=1)

            return res["hits"][0]["_score"]

        score_1 = get_score({
            "text_field": "A rider is riding a horse jumping over the barrier.",
            # "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
        })

        score_3 = get_score({
            "combo_text_image": {
            "text_field" : "A rider is riding a horse jumping over the barrier.",
            "image_field" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                },
        })

        self.assertEqual(score_1, score_3)

    def test_multimodal_tensor_combination_vectorise_call(self):
        """check if the chunks are properly created in the add_documents"""
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        def pass_through_multimodal(*arg, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the call list
            via mock
            """
            return vectorise_multimodal_combination_field(*arg, **kwargs)

        mock_multimodal_combination = unittest.mock.MagicMock()
        mock_multimodal_combination.side_effect = pass_through_multimodal

        @unittest.mock.patch("marqo.tensor_search.tensor_search.vectorise_multimodal_combination_field",
                             mock_multimodal_combination)
        def run():
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                {
                    "combo_text_image": {
                        "text_field": "A rider is riding a horse jumping over the barrier.",
                        "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    },
                    "_id": "123",
                },

                {
                    "combo_text_image": {
                        "text_field" : "test-text-two.",
                        "image_field":"https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                    },
                    "_id": "234",
                },

                {  # a normal doc
                    "combo_text_image_test": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                    "_id": "534",
                }
            ], mappings = {"combo_text_image" : {"type":"multimodal_combination",
            "weights":{"image_field": 0.5,"text_field": 0.5}}}, auto_refresh=True)

            # first multimodal-doc
            real_fied_0, field_content_0 = [call_args for call_args, call_kwargs
                                            in mock_multimodal_combination.call_args_list][0][0:2]
            assert real_fied_0 == "combo_text_image"
            assert field_content_0 ==    {
                        "text_field": "A rider is riding a horse jumping over the barrier.",
                        "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    }

            # second multimodal=doc
            real_fied_1, field_content_1 = [call_args for call_args, call_kwargs
                                            in mock_multimodal_combination.call_args_list][1][0:2]
            assert real_fied_1 == "combo_text_image"
            assert field_content_1 =={
                        "text_field" : "test-text-two.",
                        "image_field":"https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                    }
            # ensure we only call multimodal-combination twice
            assert len(mock_multimodal_combination.call_args_list) == 2

            assert json.loads(requests.get(url = f"{self.endpoint}/{self.index_name_1}/_doc/123", verify=False).text)["found"] == True
            assert json.loads(requests.get(url = f"{self.endpoint}/{self.index_name_1}/_doc/234", verify=False).text)["found"] == True
            assert json.loads(requests.get(url = f"{self.endpoint}/{self.index_name_1}/_doc/534", verify=False).text)["found"] == True

            return True
        assert run()

    def test_multimodal_field_content_dictionary_validation(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        # invalid field_content int
        res_0 = tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "A rider is riding a horse jumping over the barrier." : 0.5,
                    "image_field" : 0.5,
                },
                "_id": "123",
            }, ], mappings=self.mappings, auto_refresh=True)
        assert res_0["errors"]
        assert not json.loads(requests.get(url = f"{self.endpoint}/{self.index_name_1}/_doc/123", verify=False).text)["found"]

        try:
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="123")
            raise AssertionError
        except DocumentNotFoundError:
            pass

        # invalid field content dict
        res_1 = tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "text_field": "A rider is riding a horse jumping over the barrier.",
                    "image_field": {"image_url" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                   },
                },
                "_id": "123",
            }, ], mappings=self.mappings, auto_refresh=True)
        assert res_1["errors"]
        assert not json.loads(requests.get(url = f"{self.endpoint}/{self.index_name_1}/_doc/123", verify=False).text)["found"]
        try:
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="123")
            raise AssertionError
        except DocumentNotFoundError:
            pass

        # invalid field name format
        res_2 = tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "text_field" : "A rider is riding a horse jumping over the barrier.",
                    934343 : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",

                },
                "_id": "123",
            }, ], mappings = self.mappings, auto_refresh=True)
        assert res_2["errors"]
        assert not json.loads(requests.get(url = f"{self.endpoint}/{self.index_name_1}/_doc/123", verify=False).text)["found"]
        try:
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="123")
            raise AssertionError
        except DocumentNotFoundError:
            pass

    def test_validate_dict(self):
        test_mappings = {"my_combo_field":{"type":"multimodal_combination", "weights":{
            "test_1":0.5, "test_2":0.5
        }}}
        field = "my_combo_field"
        valid_dict = {"test_1": "test", "test_2": "test_test"}

        # valid_dict
        validate_dict(field, valid_dict, is_non_tensor_field=False, mappings=test_mappings)

        # invalid str:str format
        # str:list
        try:
            validate_dict(field, {"test_1": ["my","test"], "test_2": "test_test"}, is_non_tensor_field=False, mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "is not of valid content type" in e.message
        # str:tuple
        try:
            validate_dict(field, {"test_1": ("my","test"), "test_2": "test_test"}, is_non_tensor_field=False,
                          mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "is not of valid content type" in e.message

        # str:dict
        try:
            validate_dict(field, {"test_1": {"my":"test"}, "test_2": "test_test"}, is_non_tensor_field=False,
                          mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "is not of valid content type" in e.message

        # str:int
        try:
            validate_dict(field, {"test_1": 53213, "test_2": "test_test"}, is_non_tensor_field=False,
                          mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "is not of valid content type" in e.message

        # str:None
        try:
            validate_dict(field, {"test_1": None, "test_2": "test_test"}, is_non_tensor_field=False,
                          mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "is not of valid content type" in e.message

        # mapping is None
        try:
            validate_dict(field, valid_dict, is_non_tensor_field=False, mappings=None)
            raise AssertionError
        except InvalidArgError as e:
            assert "the parameter `mappings`" in e.message

        # field not in mappings
        try:
            validate_dict('void_field', valid_dict, is_non_tensor_field=False, mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "not in the add_document parameter mappings" in e.message

        # sub_fields not in mappings["weight"]
        try:
            validate_dict(field, {"test_void": "test", "test_2": "test_test"}, is_non_tensor_field=False, mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "Each sub_field requires a weights" in e.message

        # length of fields
        try:
            validate_dict(field, {}, is_non_tensor_field=False, mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "it must contain at least 1 field" in e.message

        # nontensor_field
        try:
            validate_dict(field, valid_dict, is_non_tensor_field=True, mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "It CANNOT be a `non_tensor_field`" in e.message

    def test_batched_vectorise_call(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        def pass_through_vectorise(*arg, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the call list
            via mock
            """
            return vectorise(*arg, **kwargs)

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @unittest.mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                {
                    "combo_text_image": {
                        "text_0": "A rider is riding a horse jumping over the barrier_0.",
                        "text_1":"A rider is riding a horse jumping over the barrier_1.",
                        "text_2":"A rider is riding a horse jumping over the barrier_2.",
                        "text_3":"A rider is riding a horse jumping over the barrier_3.",
                        "text_4":"A rider is riding a horse jumping over the barrier_4.",
                        "image_0" :  "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg",
                        "image_1" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                        "image_2" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                        "image_3" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg",
                        "image_4" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                    },
                    "_id": "111",
                },

            ], mappings = {"combo_text_image" :{"type":"multimodal_combination", "weights":{
                "text_0" : 0.1, "text_1" : 0.1, "text_2" : 0.1, "text_3" : 0.1, "text_4" : 0.1,
                "image_0" : 0.1,"image_1" : 0.1,"image_2" : 0.1,"image_3" : 0.1,"image_4" : 0.1,
            }}}, auto_refresh=True)
            # Ensure the doc is added
            assert tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="111")
            # Ensure that vectorise is only called twice
            assert len(mock_vectorise.call_args_list) == 2

            text_content = [f"A rider is riding a horse jumping over the barrier_{i}." for i in range(5)]

            real_text_content = [call_kwargs['content'] for call_args, call_kwargs
                                 in mock_vectorise.call_args_list][0]

            # Ensure the text vectorise is expected
            self.assertEqual(real_text_content, text_content)
            return True

        assert run()

    def test_batched_vectorise_call_infer_image_is_false(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: False,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        def pass_through_vectorise(*arg, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the call list
            via mock
            """
            return vectorise(*arg, **kwargs)

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @unittest.mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                {
                    "combo_text_image": {
                        "text_0": "A rider is riding a horse jumping over the barrier_0.",
                        "text_1":"A rider is riding a horse jumping over the barrier_1.",
                        "text_2":"A rider is riding a horse jumping over the barrier_2.",
                        "text_3":"A rider is riding a horse jumping over the barrier_3.",
                        "text_4":"A rider is riding a horse jumping over the barrier_4.",
                        "image_0" :  "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg",
                        "image_1" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                        "image_2" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                        "image_3" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg",
                        "image_4" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                    },
                    "_id": "111",
                },

            ], mappings = {"combo_text_image" :{"type":"multimodal_combination", "weights":{
                "text_0" : 0.1, "text_1" : 0.1, "text_2" : 0.1, "text_3" : 0.1, "text_4" : 0.1,
                "image_0" : 0.1,"image_1" : 0.1,"image_2" : 0.1,"image_3" : 0.1,"image_4" : 0.1,
            }}}, auto_refresh=True)
            # Ensure the doc is added
            assert tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="111")
            # Ensure that vectorise is only called twice
            assert len(mock_vectorise.call_args_list) == 1

            text_content = [f"A rider is riding a horse jumping over the barrier_{i}." for i in range(5)]
            text_content = text_content  + [f"https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image{i}.jpg" for i in range(5)]

            real_text_content = [call_kwargs['content'] for call_args, call_kwargs
                                 in mock_vectorise.call_args_list][0]

            # Ensure the text vectorise is expected
            self.assertEqual(real_text_content, text_content)
            return True

        assert run()

    def test_concurrent_image_downloading(self):

        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        def pass_through_load_image_from_path(*arg, **kwargs):
            return load_image_from_path(*arg, **kwargs)

        mock_load_image_from_path = unittest.mock.MagicMock()
        mock_load_image_from_path.side_effect = pass_through_load_image_from_path

        @unittest.mock.patch("marqo.s2_inference.clip_utils.load_image_from_path", mock_load_image_from_path)
        def run():
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                {
                    "combo_text_image": {
                        "text_0": "A rider is riding a horse jumping over the barrier_0.",
                        "text_1": "A rider is riding a horse jumping over the barrier_1.",
                        "text_2": "A rider is riding a horse jumping over the barrier_2.",
                        "text_3": "A rider is riding a horse jumping over the barrier_3.",
                        "text_4": "A rider is riding a horse jumping over the barrier_4.",
                        "image_0": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg",
                        "image_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                        "image_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                        "image_3": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg",
                        "image_4": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                    },
                    "_id": "111",
                },

            ], mappings={"combo_text_image": {"type": "multimodal_combination", "weights": {
                "text_0": 0.1, "text_1": 0.1, "text_2": 0.1, "text_3": 0.1, "text_4": 0.1,
                "image_0": 0.1, "image_1": 0.1, "image_2": 0.1, "image_3": 0.1, "image_4": 0.1,
            }}}, auto_refresh=True)
            assert tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="111")
            # Ensure that vectorise is only called twice
            assert len(mock_load_image_from_path.call_args_list) == 5

            return True

        assert run()

    def test_lexical_search_on_multimodal_combination(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "random/small",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {"Title": "Extravehicular Mobility Unit (EMU)",
             "Description": "The EMU is a spacesuit that provides environmental protection",
             "_id": "article_591",
             "Genre": "Science",
             "my_combination_field": {
                 "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                 "some_text": "hello there",
                 "lexical_field": "search me please",}},],
                                    mappings={
                                        "my_combination_field": {
                                            "type": "multimodal_combination",
                                            "weights": {
                                                "my_image": 0.5,
                                                "some_text": 0.5,
                                                "lexical_field": 0.1,
                                                "additional_field" : 0.2,
                                            }
                                        }}
                                    , auto_refresh=True)

        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {"Title": "text",
             "Description": "text_2",
             "_id": "article_592",
             "Genre": "text",
             "my_combination_field": {
                 "my_image_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                 "some_text_1": "hello there",
                 "lexical_field_1": "no no no",
                 "additional_field_1" : "test_search here"}}
            ],
            mappings={
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "my_image_1": 0.5,
                        "some_text_1": 0.5,
                        "lexical_field_1": 0.1,
                        "additional_field_1" : 0.2,
                    }
                }}
                ,auto_refresh=True)


        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text="search me please")
        assert res["hits"][0]["_id"] == "article_591"

        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text="test_search here")
        assert res["hits"][0]["_id"] == "article_592"

    def test_overwrite_multimodal_tensor_field(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "random/small",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {"Title": "Extravehicular Mobility Unit (EMU)",
             "Description": "The EMU is a spacesuit that provides environmental protection",
             "_id": "article_591",
             "Genre": "Science",
             "my_combination_field": "dummy"
             },]
                                    , auto_refresh=True)

        try:
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                {"Title": "text",
                 "Description": "text_2",
                 "_id": "article_592",
                 "Genre": "text",
                 "my_combination_field": {
                     "my_image_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                     "some_text_1": "hello there",
                     "lexical_field_1": "no no no",
                     "additional_field_1" : "test_search here"}}
                ],
                mappings={
                    "my_combination_field": {
                        "type": "multimodal_combination",
                        "weights": {
                            "my_image_1": 0.5,
                            "some_text_1": 0.5,
                            "lexical_field_1": 0.1,
                            "additional_field_1" : 0.2,
                        }
                    }}
                    ,auto_refresh=True)

            raise AssertionError
        except MarqoWebError:
            pass

    def test_search_with_filtering_and_infer_image_false(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: False,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"Title": "Extravehicular Mobility Unit (EMU)",
                 "_id": "0",
                 "my_combination_field": {
                     "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                     "some_text": "hello there",
                     "filter_field": "test_this_0", }},

                {"Title": "Extravehicular Mobility Unit (EMU)",
                 "_id": "1",
                 "my_combination_field": {
                     "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                     "some_text": "hello there",
                     "filter_field": "test_this_1", }},

                {"Title": "Extravehicular Mobility Unit (EMU)",
                 "_id": "2",
                 "my_combination_field": {
                     "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                     "some_text": "hello there",
                     "filter_field": "test_this_2", }},
            ],
            mappings={
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "my_image": 0.5,
                        "some_text": 0.5,
                        "filter_field": 0,
                    }
                }}
            , auto_refresh=True)

        res_exist_0 = tensor_search.search(index_name=self.index_name_1, config=self.config,
                                           text = "", filter="my_combination_field.filter_field: test_this_0")

        assert res_exist_0["hits"][0]["_id"] == "0"

        res_exist_2 = tensor_search.search(index_name=self.index_name_1, config=self.config,
                                           text="", filter="my_combination_field.filter_field: test_this_2")

        assert res_exist_2["hits"][0]["_id"] == "2"

        res_nonexist_1 = tensor_search.search(index_name=self.index_name_1, config=self.config,
                                           text="", filter="my_combination_field.filter_field: test_this_5")

        assert res_nonexist_1["hits"] == []

    def test_index_info_cache_update(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"Title": "Extravehicular Mobility Unit (EMU)",
                 "_id": "0",
                 "my_combination_field": {
                     "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                     "some_text": "hello there",
                     "filter_field": "test_this_0", }},

                {"Title": "what is this",
                 "_id": "1",
                 "my_combination_field": {
                     "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                     "some_text": "hello there",
                     "filter_field": "test_this_1", }},

                {"Title": "have a test",
                 "_id": "2",
                 "my_combination_field": {
                     "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                     "some_text": "hello there",
                     "filter_field": "test_this_2", }},
            ],
            mappings={
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "my_image": 0.5,
                        "some_text": 0.5,
                        "filter_field": 0,
                    }
                }}
            , auto_refresh=True)

        pre_res_0 = tensor_search.search(index_name=self.index_name_1, config=self.config,
                                           text = "", filter="my_combination_field.filter_field: test_this_0")
        pre_res_1 = tensor_search.search(index_name=self.index_name_1, config=self.config,
                                         text="hello there")
        pre_res_2 = tensor_search._lexical_search(index_name=self.index_name_1, config=self.config,text="have a test")

        index_info = tensor_search.get_index_info(config=self.config, index_name=self.index_name_1)

        post_res_0 = tensor_search.search(index_name=self.index_name_1, config=self.config,
                                         text="", filter="my_combination_field.filter_field: test_this_0")
        post_res_1 = tensor_search.search(index_name=self.index_name_1, config=self.config,
                                         text="hello there")
        post_res_2 = tensor_search._lexical_search(index_name=self.index_name_1, config=self.config, text="have a test")

        assert pre_res_2["hits"] == post_res_2["hits"]
        assert pre_res_1["hits"] == post_res_1["hits"]
        assert pre_res_0["hits"] == post_res_0["hits"]

    def test_duplication_in_child_fields(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "random/small",
                    IndexSettingsField.treat_urls_and_pointers_as_images: False,
                    IndexSettingsField.normalize_embeddings: True
                }
            })

        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {"Title": "Extravehicular Mobility Unit (EMU)",
             "Description": "The EMU is a spacesuit that provides environmental protection",
             "_id": "article_591",
             "Genre": "Science",
             "my_combination_field_0": {
                 "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                 "some_text": "hello there",
                 "lexical_field": "search me please", }}, ],
                                    mappings={
                                        "my_combination_field_0": {
                                            "type": "multimodal_combination",
                                            "weights": {
                                                "my_image": 0.5,
                                                "some_text": 0.5,
                                                "lexical_field": 0.1,
                                            }
                                        }}
                                    , auto_refresh=True)

        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {"Title": "text",
             "Description": "text_2",
             "_id": "article_592",
             "Genre": "text",
             "my_combination_field_1": {
                 "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                 "some_text": "marqo is good",
                 "lexical_field": "no no no",
                 "additional_field": "i can hear you"}}
        ],
                                    mappings={
                                        "my_combination_field_1": {
                                            "type": "multimodal_combination",
                                            "weights": {
                                                "my_image": 0.5,
                                                "some_text": 0.5,
                                                "lexical_field": 0.1,
                                                "additional_field": 0.2,
                                            }
                                        }}
                                    , auto_refresh=True)

        true_text_fields = tensor_search.get_index_info(self.config, index_name=self.index_name_1).get_true_text_properties()
        # 3 from multimodal_field_0, 4 from multimodal_field_1, 3 common fields
        assert len(true_text_fields) == 10

        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text="hello there")
        assert res["hits"][0]["_id"] == "article_591"

        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text="marqo is good")
        assert res["hits"][0]["_id"] == "article_592"

    def test_multimodal_combination_open_search_chunks(self):
        test_doc ={
                 "my_combination_field": {
                     "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                     "text": "marqo is good" },
                   "_id": "123",
                   }
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "random/small",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: True
                }
            })

        res = tensor_search.add_documents(
            self.config,
            docs = [test_doc],
            auto_refresh=True, index_name=self.index_name_1,
            mappings={"my_combination_field": {"type":"multimodal_combination", "weights":{
                "text":0.5, "image":0.5
            }}}
        )

        doc_w_facets = tensor_search.get_document_by_id(
            self.config, index_name=self.index_name_1, document_id='123', show_vectors=True)
        # check tensor facets:
        assert len(doc_w_facets[TensorField.tensor_facets]) == 1
        assert 'my_combination_field' in doc_w_facets[TensorField.tensor_facets][0]

        assert doc_w_facets['my_combination_field'] == test_doc['my_combination_field']
        assert doc_w_facets[TensorField.tensor_facets][0]['my_combination_field'] == json.dumps(test_doc['my_combination_field'])

        # check OpenSearch, to ensure the list got added as a filter field
        original_doc = requests.get(
            url=F"{self.endpoint}/{self.index_name_1}/_doc/123",
            verify=False
        ).json()
        assert len(original_doc['_source']['__chunks']) == 1
        multi_combo_chunk = original_doc['_source']['__chunks'][0]
        # check if the chunk represents the tensorsied "multimodal_combination" field
        assert multi_combo_chunk['__field_name'] == 'my_combination_field'
        assert multi_combo_chunk['__field_content'] == json.dumps(test_doc['my_combination_field'])
        assert isinstance(multi_combo_chunk['__vector_my_combination_field'], list)
        # Check if all filter fields are  there (inc. the non tensorised my_list):
        assert multi_combo_chunk['my_combination_field'] == test_doc["my_combination_field"]

        # # check index info. child fields needs to be keyword within each chunk
        index_info = tensor_search.backend.get_index_info(config=self.config, index_name=self.index_name_1)
        assert index_info.properties['my_combination_field']['properties']['image']["type"] == 'text'
        assert index_info.properties['my_combination_field']['properties']['text']["type"] == 'text'
        assert index_info.properties['__chunks']['properties']['my_combination_field']['properties']['text']["type"]   == 'keyword'
        assert index_info.properties['__chunks']['properties']['my_combination_field']['properties']['image']["type"] == 'keyword'
        assert index_info.properties['__chunks']['properties']['__vector_my_combination_field']['type']  == 'knn_vector'

    def test_multimodal_child_fields_order(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: True
                }
            })

        doc ={
                "combo_text_image": {
                    "text_field_1": "A rider is riding a horse jumping over the barrier.",
                    "text_field_2": "What is the best to wear on the moon?",
                    "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                },
            }

        doc_1 = {
                "combo_text_image": {
                    "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    "text_field_1": "A rider is riding a horse jumping over the barrier.",
                    "text_field_2": "What is the best to wear on the moon?",
                    "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                },
            }

        doc_2 = {
                "combo_text_image": {
                    "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                    "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    "text_field_1": "A rider is riding a horse jumping over the barrier.",
                    "text_field_2": "What is the best to wear on the moon?",
                },
            }

        doc_3 = {
                "combo_text_image": {
                    "text_field_1": "A rider is riding a horse jumping over the barrier.",
                    "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                    "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    "text_field_2": "What is the best to wear on the moon?",
                },
            }

        with patch("numpy.mean", wraps=np.mean) as mock_mean:
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                doc, doc_1, doc_2, doc_3
            ], mappings={"combo_text_image": {"type": "multimodal_combination",
                                              "weights": {"image_field_1": 0.2, "image_field_2": -1, "text_field_1": 0.38, "text_field_2": 0}}}, auto_refresh=True)

            args_list = [args[0] for args in mock_mean.call_args_list]

        combined_tensor = np.squeeze(np.mean(args_list[0][0], axis = 0))

        permuted_tensor_1 = np.squeeze(np.mean(args_list[1][0], axis = 0))
        permuted_tensor_2 = np.squeeze(np.mean(args_list[2][0], axis=0))
        permuted_tensor_3 = np.squeeze(np.mean(args_list[3][0], axis=0))

        assert np.allclose(combined_tensor, permuted_tensor_1, atol=1e-9)
        assert np.allclose(combined_tensor, permuted_tensor_2, atol=1e-9)
        assert np.allclose(combined_tensor, permuted_tensor_3, atol=1e-9)

    def test_multimodal_child_fields_order_from_os(self):
        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        doc = {
            "_id":"d0",
            "combo_text_image": {
                "text_field_1": "A rider is riding a horse jumping over the barrier.",
                "text_field_2": "What is the best to wear on the moon?",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
            },
        }

        doc_1 = {
            "_id": "d1",
            "combo_text_image": {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "text_field_1": "A rider is riding a horse jumping over the barrier.",
                "text_field_2": "What is the best to wear on the moon?",
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
            },
        }

        doc_2 = {
            "_id": "d2",
            "combo_text_image": {
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "text_field_1": "A rider is riding a horse jumping over the barrier.",
                "text_field_2": "What is the best to wear on the moon?",
            },
        }

        doc_3 = {
            "_id": "d3",
            "combo_text_image": {
                "text_field_1": "A rider is riding a horse jumping over the barrier.",
                "image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "text_field_2": "What is the best to wear on the moon?",
            },
        }

        with patch("numpy.mean", wraps=np.mean) as mock_mean:
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                doc, doc_1, doc_2, doc_3
            ], mappings={"combo_text_image": {"type": "multimodal_combination",
                                              "weights": {"image_field_1": 0.2, "image_field_2": -1,
                                                          "text_field_1": 0.38, "text_field_2": 0}}}, auto_refresh=True)
            docs = tensor_search.get_documents_by_ids(
                config=self.config, document_ids=["d0", "d1", "d2", "d3"],
                index_name=self.index_name_1, show_vectors=True)

            args_list = [args[0] for args in mock_mean.call_args_list]

        combined_tensor = np.squeeze(np.mean(args_list[0][0], axis=0))
        os0, os1, os2, os3 = [res["_tensor_facets"][0]["_embedding"] for res in docs['results']]

        assert np.allclose(combined_tensor, os0, atol=1e-9)
        assert np.allclose(combined_tensor, os1, atol=1e-9)
        assert np.allclose(combined_tensor, os2, atol=1e-9)
        assert np.allclose(combined_tensor, os3, atol=1e-9)
