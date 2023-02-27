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


class TestMultimodalTensorCombination(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.mappings = {"combo_text_image" :{"type": "multimodal_combination", "weights" : {
            "text" : 0.5, "image" : 0.8}
        }}
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
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

    def test_multimodal_tensor_combiantion_tensor_value(self):
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
                "text_field" : "A rider is riding a horse jumping over the barrier.",
                "image_field" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                    },

                "_id": "0"
            },

            {
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "_id": "1"
            },
            {
                "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "_id": "2"
            },

        ], auto_refresh=True, mappings = {"combo_text_image" : {"type":"multimodal_combination",
            "weights":{"image_field": 1,"text_field": -1}}})

        combo_tensor = np.array(tensor_search.get_document_by_id(config=self.config,
                                                                 index_name=self.index_name_1, document_id="0",
                                                                 show_vectors=True)['_tensor_facets'][0]["_embedding"])
        text_tensor = \
            np.array(tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="1",
                                                      show_vectors=True)['_tensor_facets'][0]["_embedding"])
        image_tensor = \
            np.array(tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="2",
                                                      show_vectors=True)['_tensor_facets'][0]["_embedding"])

        assert np.sum(combo_tensor - (text_tensor * -1 + image_tensor * 1)) == 0

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
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "A rider is riding a horse jumping over the barrier." : 0.5,
                    "image_field" : 0.5,
                },
                "_id": "123",
            }, ], mappings=self.mappings, auto_refresh=True)
        try:
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="123")
            raise AssertionError
        except DocumentNotFoundError:
            pass

        # invalid field content dict
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "text_field": "A rider is riding a horse jumping over the barrier.",
                    "image_field": {"image_url" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                   },
                },
                "_id": "123",
            }, ], mappings=self.mappings, auto_refresh=True)
        try:
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="123")
            raise AssertionError
        except DocumentNotFoundError:
            pass

        # invalid field name format
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "text_field" : "A rider is riding a horse jumping over the barrier.",
                    934343 : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",

                },
                "_id": "123",
            }, ], mappings = self.mappings, auto_refresh=True)
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
            validate_dict(field, {"test_void": "test"}, is_non_tensor_field=False, mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "it must contain at least 2 fields" in e.message

        # nontensor_field
        try:
            validate_dict(field, valid_dict, is_non_tensor_field=True, mappings=test_mappings)
            raise AssertionError
        except InvalidArgError as e:
            assert "It CAN NOT be a `non_tensor_field`" in e.message



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
                 "lexical_field": "search me please", }},],
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
                 "my_image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                 "some_text": "hello there",
                 "lexical_field": "no no no",
                 "additional_field" : "search me again"}}
            ],
                                    mappings={
                                        "my_combination_field": {
                                            "type": "multimodal_combination",
                                            "weights": {
                                                "my_image": 0.5,
                                                "some_text": 0.5,
                                                "lexical_field": 0.1,
                                                "additional_field" : 0.2,

                                                # "my_combonitaion_field.lexical_field"
                                            }
                                        }}
                                    , auto_refresh=True)

        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text="search me please")
        assert res["hits"][0]["_id"] == "article_591"

        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1, text="search me again")
        assert res["hits"][0]["_id"] == "article_592"

        index_info = tensor_search.get_index_info(config=self.config, index_name=self.index_name_1).get_true_text_properties()




        #pprint.pprint(index_info)
        # doc = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="article_591")
        # # index_info = tensor_search.get_index_info(config=self.config, index_name=self.index_name_1)
        # # pprint.pprint(doc)
        # self.endpoint = self.authorized_url
        # print(res["hits"][0])
        #pprint.pprint(marqo.tensor_search.backend.get_index_info(config=self.config, index_name=self.index_name_1).get_true_text_properties())

        #
        # pprint.pprint(json.loads(requests.get(url =
        #                                       f"{self.endpoint}/{self.index_name_1}/_mapping", verify=False).text))



        # print("searchable fields")
        # pprint.pprint(tensor_search.get_index_info(
        #     config=self.config, index_name=self.index_name_1
        # ).get_true_text_properties())
















    # added_doc = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="0",
    #                                              show_vectors=True)
    # for key, value in expected_doc.items():
    #     if not isinstance(value, dict):
    #         assert expected_doc[key] == added_doc[key]
    #     else:
    #         assert list(expected_doc[key]) == added_doc[key]
    #
    #
    # tensor_field = added_doc["_tensor_facets"]
    # self.assertEqual(len(tensor_field), 2)
    # # for "Title" : "Horse Rider"
    # assert "_embedding" in tensor_field[0]
    # assert tensor_field[0]["Title"] == expected_doc["Title"]
    #
    # # for combo filed
    # assert "_embedding" in tensor_field[1]
    # assert tensor_field[1]["combo_text_image"] == list(expected_doc["combo_text_image"])
