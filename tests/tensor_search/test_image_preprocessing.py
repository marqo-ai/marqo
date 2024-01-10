import os
from unittest import mock

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestImagePreprocessing(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_image_index = cls.structured_marqo_index_request(
            model=Model(name="random/small"),
            fields=[
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer)
            ],
            tensor_fields=["image_field_1", "image_field_2"],
            image_preprocessing=ImagePreProcessing(patchMethod="simple"),
        )

        unstructured_image_index = cls.unstructured_marqo_index_request(
            model=Model(name="random/small"),
            treat_urls_and_pointers_as_images=True,
            image_preprocessing=ImagePreProcessing(patchMethod="frcnn"),
        )

        cls.indexes = cls.create_indexes([structured_image_index, unstructured_image_index])

        cls.structured_image_index = structured_image_index.name
        cls.unstructured_image_index = unstructured_image_index.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_image_preprocess_search_highlights_format(self):
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg"

        documents = [{"image_field_1": image_url, "_id": "1"}]

        for index_name in [self.structured_image_index, self.unstructured_image_index]:
            tensor_fields = None if index_name == self.structured_image_index else ["image_field_1"]

            with self.subTest(f"index_name = {index_name}"):
                tensor_search.add_documents(config=self.config,
                                            add_docs_params=AddDocsParams(index_name=index_name,
                                                                          docs=documents,
                                                                          tensor_fields=tensor_fields))
                search_result = tensor_search.search(config=self.config,
                                                     index_name=index_name,
                                                     text="test")
                self.assertIn("_highlights", search_result["hits"][0])
                self.assertIn("image_field_1", search_result["hits"][0]["_highlights"])
                self.assertTrue(isinstance(search_result["hits"][0]["_highlights"]["image_field_1"], str))
                self.assertTrue(isinstance(eval(search_result["hits"][0]["_highlights"]["image_field_1"]), list))
                self.assertEqual(4, len(eval(search_result["hits"][0]["_highlights"]["image_field_1"])))

    def test_image_preprocess_get_documents_format(self):
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg"

        documents = [{"image_field_1": image_url, "_id": "1"}]

        for index_name in [self.structured_image_index, self.unstructured_image_index]:
            tensor_fields = None if index_name == self.structured_image_index else ["image_field_1"]

            with self.subTest(f"index_name = {index_name}"):
                tensor_search.add_documents(config=self.config,
                                            add_docs_params=AddDocsParams(index_name=index_name,
                                                                          docs=documents,
                                                                          tensor_fields=tensor_fields))
                get_doc_result = tensor_search.get_document_by_id(config=self.config,
                                                                  index_name=index_name,
                                                                  document_id="1",
                                                                  show_vectors=True)
                for tensor_facet in get_doc_result["_tensor_facets"]:
                    self.assertIn("image_field_1", tensor_facet)
                    self.assertTrue(isinstance(tensor_facet["image_field_1"], str))
                    self.assertTrue(isinstance(eval(tensor_facet["image_field_1"]), list))
                    self.assertEqual(4, len(eval(tensor_facet["image_field_1"])))