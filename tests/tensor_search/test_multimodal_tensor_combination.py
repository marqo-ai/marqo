import unittest.mock

from marqo.errors import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.tensor_search import add_documents, vectorise_multimodal_combination_field
from marqo.errors import DocumentNotFoundError
import numpy as np


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
                            IndexSettingsField.normalize_embeddings:False
                        }
                    })

        tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "Title": "Horse rider",
                "combo_text_image": {
                    "A rider is riding a horse jumping over the barrier." : {
                        "weight" : 0.5,
                                        },
                    "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                    "weight": 0.5,
                                        },
                },
                "_id": "0"
            },

            {
                "Title": "Horse rider",
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "image_field":"https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "_id": "1"
            },
        ], auto_refresh=True)

        expected_doc = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id="0")
        print(expected_doc)
        self.assertEqual(expected_doc, dict({
                "Title": "Horse rider",
                "combo_text_image": {
                    "A rider is riding a horse jumping over the barrier." : {
                        "weight" : 0.5,
                                        },
                    "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg": {
                    "weight": 0.5,
                                        },
                },
                "_id": "0"
            },) )


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

            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[document], auto_refresh=True)
            self.assertEqual(1, tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfDocuments"])
            res = tensor_search.search(config=self.config, index_name=self.index_name_1,
                                       text="", result_count=1)

            return res["hits"][0]["_score"]

        score_1 = get_score({
                "text_field": "A rider is riding a horse jumping over the barrier.",
                #"image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
            })


        score_2 = get_score({
                #"text_field": "A rider is riding a horse jumping over the barrier.",
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

        assert (score_3 >= min(score_1, score_2)) and (score_3 <= max(score_1,score_2))


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
                             index_name=self.index_name_1,document_id="0",show_vectors=True )['_tensor_facets'][0]["_embedding"])
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
                #"image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
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

        self.assertEqual(score_1,score_3)


    def test_multimodal_tensor_combination_vectorise_call(self):
        """check if the chunks are properly created in the add_documents"""

        tensor_search.create_vector_index(
                        index_name=self.index_name_1, config=self.config, index_settings={
                        IndexSettingsField.index_defaults: {
                            IndexSettingsField.model: "ViT-B/32",
                            IndexSettingsField.treat_urls_and_pointers_as_images: True,
                            IndexSettingsField.normalize_embeddings:False
                        }
                    })

        def pass_through_multimodal(*arg, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the call list
            via mock
            """
            return vectorise_multimodal_combination_field(*arg, **kwargs)

        mock_multimodal_combination = unittest.mock.MagicMock()
        mock_multimodal_combination.side_effect = pass_through_multimodal
        @unittest.mock.patch("marqo.tensor_search.tensor_search.vectorise_multimodal_combination_field", mock_multimodal_combination)
        def run():
            tensor_search.add_documents(config=self.config, index_name=self.index_name_1, docs = [
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

                {   # a normal doc
                    "combo_text_image_test": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                    "_id": "534",
                }
            ], auto_refresh=True)

            # first multimodal-doc
            real_fied_0, field_content_0 = [call_args for call_args, call_kwargs
                                        in  mock_multimodal_combination.call_args_list][0][0:2]
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
                                        in  mock_multimodal_combination.call_args_list][1][0:2]
            assert real_fied_1 == "combo_text_image_test"
            assert field_content_1 == dict({
                        "test-text-two.": {
                            "weight": 1,
                        },
                        "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg": {
                            "weight": 0,
                        },
                    },)

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
                            IndexSettingsField.normalize_embeddings:False
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
            },], auto_refresh=True)
        try:
            tensor_search.get_document_by_id(config=self.config,index_name=self.index_name_1, document_id="123")
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

