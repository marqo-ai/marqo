import unittest.mock

from marqo.errors import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.tensor_search import add_documents, vectorise_multimodal_combination_field
from marqo.errors import DocumentNotFoundError
import numpy as np
from marqo.tensor_search.validation import validate_dict
from marqo.s2_inference.s2_inference import vectorise
from marqo.s2_inference.clip_utils import load_image_from_path



class TestMultimodalTensorCombination(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
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
                "A rider is riding a horse jumping over the barrier.": {
                    "weight": 0.5,
                },
                "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                    "weight": 0.5,
                },
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
        ], auto_refresh=True)

        added_doc = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="0",
                                                     show_vectors=True)
        for key, value in expected_doc.items():
            if not isinstance(value, dict):
                assert expected_doc[key] == added_doc[key]
            else:
                assert list(expected_doc[key]) == added_doc[key]


        tensor_field = added_doc["_tensor_facets"]
        self.assertEqual(len(tensor_field), 2)
        # for "Title" : "Horse Rider"
        assert "_embedding" in tensor_field[0]
        assert tensor_field[0]["Title"] == expected_doc["Title"]

        # for combo filed
        assert "_embedding" in tensor_field[1]
        assert tensor_field[1]["combo_text_image"] == list(expected_doc["combo_text_image"])

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
                                        auto_refresh=True)
            self.assertEqual(1, tensor_search.get_stats(config=self.config, index_name=self.index_name_1)[
                "numberOfDocuments"])
            res = tensor_search.search(config=self.config, index_name=self.index_name_1,
                                       text="", result_count=1)

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
                "A rider is riding a horse jumping over the barrier.": {
                    "weight": 0.5,
                },
                "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                    "weight": 0.5,
                },
            }
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
                    "A rider is riding a horse jumping over the barrier.": {
                        "weight": -1,
                    },
                    "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                        "weight": 1,
                    },
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

        ], auto_refresh=True)

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

    def test_multimodal_tensor_combination_weight(self):

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
                                        auto_refresh=True)
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
                "A rider is riding a horse jumping over the barrier.": {
                    "weight": 1,
                },
                "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                    "weight": 0,
                },
            }
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
                        "A rider is riding a horse jumping over the barrier.": {
                            "weight": 1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                            "weight": 0,
                        },
                    },
                    "_id": "123",
                },

                {
                    "combo_text_image_test": {
                        "test-text-two.": {
                            "weight": 1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg": {
                            "weight": 0,
                        },
                    },
                    "_id": "234",
                },

                {  # a normal doc
                    "combo_text_image_test": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                    "_id": "534",
                }
            ], auto_refresh=True)

            # first multimodal-doc
            real_fied_0, field_content_0 = [call_args for call_args, call_kwargs
                                            in mock_multimodal_combination.call_args_list][0][0:2]
            assert real_fied_0 == "combo_text_image"
            assert field_content_0 == dict({
                "A rider is riding a horse jumping over the barrier.": {
                    "weight": 1,
                },
                "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                    "weight": 0,
                },
            })

            # second multimodal=doc
            real_fied_1, field_content_1 = [call_args for call_args, call_kwargs
                                            in mock_multimodal_combination.call_args_list][1][0:2]
            assert real_fied_1 == "combo_text_image_test"
            assert field_content_1 == dict({
                "test-text-two.": {
                    "weight": 1,
                },
                "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg": {
                    "weight": 0,
                },
            }, )

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

        # invalid weight string
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "A rider is riding a horse jumping over the barrier.": {
                        "weight-void": 1,
                    },
                    "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                        "weight": 0,
                    },
                },
                "_id": "123",
            }, ], auto_refresh=True)
        try:
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="123")
            raise AssertionError
        except DocumentNotFoundError:
            pass

        # invalid weight format string
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "A rider is riding a horse jumping over the barrier.": {
                        "weight-void": "1.0",
                    },
                    "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                        "weight": 0,
                    },
                },
                "_id": "123",
            }, ], auto_refresh=True)
        try:
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="123")
            raise AssertionError
        except DocumentNotFoundError:
            pass

        # invalid field content format
        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "combo_text_image": {
                    "A rider is riding a horse jumping over the barrier.": 0.5,
                    "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                        "weight": 0,
                    },
                },
                "_id": "123",
            }, ], auto_refresh=True)
        try:
            tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="123")
            raise AssertionError
        except DocumentNotFoundError:
            pass

    def test_validate_dict(self):
        valid_dict = [{"test": {"weight": 0.3}, "test_test": {"weight": -0.3}},
                      {"test": {"weight": 0.5}, "test_test": {"weight": 0.3}, "test_3": {"weight": 0.6}},
                      {"test": {"weight": 1}, "test_test":{"weight": -1}},
                      {"test": {"weight": 2}, "test_test":{"weight": -1}}]

        invalid_dict = [{"test": {"weight": 0.5}},  # field should be not less than 2
                        {3232: {"weight": 0.5}, "test_2": {"weight": 0.5}},  # field can only be string
                        {"test": {"weightsfd": 0.5}, "test_test": {"weight": 0.3}},  # incorrect spelling of "weight"
                        {"test": {"weight": "1"}, "test_test": {"weight": 0.3}},  # weight is not a float or int
                        {"test": 0.5, "test_test": {"weight": 0.3}},  # incorrect weight format
                        {"test": {"test": 0.5}, "test_test": {"weight": 0.3}},  # no weight for a field
                        {},  # empty
                        ]

        for valid_content_field in valid_dict:
            validate_dict(valid_content_field, is_non_tensor_field=False)
            try:
                validate_dict(valid_content_field, is_non_tensor_field=True)
                raise AssertionError
            except InvalidArgError as e:
                assert "non_tensor_field" in e.message
                pass

        for invalid_content_field in invalid_dict:
            try:
                validate_dict(invalid_content_field, is_non_tensor_field=False)
                raise AssertionError
            except InvalidArgError:
                pass

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
                        "A rider is riding a horse jumping over the barrier_0.": {
                            "weight": 0.1,
                        },
                        "A rider is riding a horse jumping over the barrier_1.": {
                            "weight": 0.1,
                        },
                        "A rider is riding a horse jumping over the barrier_2.": {
                            "weight": 0.1,
                        },
                        "A rider is riding a horse jumping over the barrier_3.": {
                            "weight": 0.1,
                        },
                        "A rider is riding a horse jumping over the barrier_4.": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg": {
                            "weight": 0.1,
                        },
                    },
                    "_id": "111",
                },

            ], auto_refresh=True)
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
            return load_image_from_path(*arg,**kwargs)

        mock_load_image_from_path = unittest.mock.MagicMock()
        mock_load_image_from_path.side_effect = pass_through_load_image_from_path

        @unittest.mock.patch("marqo.s2_inference.clip_utils.load_image_from_path", mock_load_image_from_path)
        def run():
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
                {
                    "combo_text_image": {
                        "A rider is riding a horse jumping over the barrier_0.": {
                            "weight": 0.1,
                        },
                        "A rider is riding a horse jumping over the barrier_1.": {
                            "weight": 0.1,
                        },
                        "A rider is riding a horse jumping over the barrier_2.": {
                            "weight": 0.1,
                        },
                        "A rider is riding a horse jumping over the barrier_3.": {
                            "weight": 0.1,
                        },
                        "A rider is riding a horse jumping over the barrier_4.": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg": {
                            "weight": 0.1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg": {
                            "weight": 0.1,
                        },
                    },
                    "_id": "111",
                },

            ], auto_refresh=True)

            assert tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="111")
            # Ensure that vectorise is only called twice
            assert len(mock_load_image_from_path.call_args_list) == 5

            return True

        assert run()

    def test_lexical_search_on_multimodal_combination(self):

        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {"title": "test_0",
             "combo_text_image_0": {
                 "A rider is riding a horse jumping over the barrier_0.": {
                     "weight": 0.1, },
                 "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg": {
                     "weight": 0.1,
                 }
             }, "_id": "0"},

            {"title": "test_1",
             "combo_text_image_1": {
                 "A rider is riding a horse jumping over the barrier_1.": {
                     "weight": 0.1, },
                 "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                     "weight": 0.1,
                 }}, "_id": "1"},

            {"title": "test_2",
             "combo_text_image_2": {
                 "A rider is riding a horse jumping over the barrier_2.": {
                     "weight": 0.1, },
                 "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg": {
                     "weight": 0.1,
                 }}, "_id": "2"},

            {"title": "test_3",
             "combo_text_image_3": {
                 "A rider is riding a horse jumping over the barrier_3.": {
                     "weight": 0.1, },
                 "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg": {
                     "weight": 0.1,
                 }}, "_id": "3"},

            {"title": "combo_4",
             "combo_text_image_4": {
                 "A rider is riding a horse jumping over the barrier_4.": {
                     "weight": 0.1, },
                 "please search me.": {
                     "weight": 0.2,
                 },
                 "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg": {
                     "weight": 0.1,
                 }}, "_id": "4"},
        ], auto_refresh=True, non_tensor_fields=["combo_text_image_4_test"])

        res = tensor_search._lexical_search(config=self.config, index_name=self.index_name_1,
                                            text="please search me")
        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfDocuments"] == 5
        assert res["hits"][0]["title"] == "combo_4"
        assert res["hits"][0]["combo_text_image_4"] == [
            "A rider is riding a horse jumping over the barrier_4.",
            "please search me.",
            "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg"
        ]


