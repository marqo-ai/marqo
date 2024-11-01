import os
from typing import cast, List
from unittest import mock

from marqo.core.exceptions import TooManyFieldsError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model, ImagePreProcessing, PatchMethod, SemiStructuredMarqoIndex
from marqo.core.semi_structured_vespa_index.semi_structured_add_document_handler import SemiStructuredFieldCountConfig
from marqo.tensor_search import tensor_search, index_meta_cache
from marqo.tensor_search.enums import SearchMethod
from tests.marqo_test import MarqoTestCase, TestImageUrls


class TestAddDocumentsSemiStructuredAddFields(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # We use a different index for each test since the test will change the index
        text_index_1 = cls.unstructured_marqo_index_request()

        text_index_2 = cls.unstructured_marqo_index_request()

        text_index_3 = cls.unstructured_marqo_index_request()

        text_index_4 = cls.unstructured_marqo_index_request()

        text_index_5 = cls.unstructured_marqo_index_request()

        text_index_6 = cls.unstructured_marqo_index_request()

        image_index_with_chunking = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.Frcnn),
            treat_urls_and_pointers_as_images=True
        )

        cls.indexes = cls.create_indexes([
            text_index_1,
            text_index_2,
            text_index_3,
            text_index_4,
            text_index_5,
            text_index_6,
            image_index_with_chunking,
        ])

        cls.text_index_1 = text_index_1.name
        cls.text_index_2 = text_index_2.name
        cls.text_index_3 = text_index_3.name
        cls.text_index_4 = text_index_4.name
        cls.text_index_5 = text_index_5.name
        cls.text_index_6 = text_index_6.name
        cls.image_index_with_chunking = image_index_with_chunking.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def _add_and_get_doc(self, index_name: str, doc_id: str, tensor_fields: List[str], use_existing_tensors=False):
        add_doc_result = self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=index_name,
                docs=[{
                    "_id": doc_id,
                    "title": "content 1",
                    "desc": "content 2. blah blah blah"
                }],
                device="cpu", tensor_fields=tensor_fields, use_existing_tensors=use_existing_tensors
            )
        )
        self.assertFalse(add_doc_result.errors)

        return tensor_search.get_document_by_id(
            config=self.config, index_name=index_name,
            document_id=doc_id, show_vectors=True
        )

    def test_add_documents_should_successfully_add_new_tensor_fields(self):
        doc1 = self._add_and_get_doc(self.text_index_1, "123", ["title"])
        self.assertEquals(1, len(doc1['_tensor_facets']))
        self.assertIn('title', doc1['_tensor_facets'][0])

        # add a second doc with different tensor_fields
        doc2 = self._add_and_get_doc(self.text_index_1, "456", ["desc"])
        self.assertEquals(1, len(doc2['_tensor_facets']))
        self.assertIn('desc', doc2['_tensor_facets'][0])

        updated_index = cast(SemiStructuredMarqoIndex, self.config.index_management.get_index(self.text_index_1))
        self.assertEqual({'title', 'desc'}, set(updated_index.tensor_field_map.keys()))

    def test_add_documents_should_override_tensor_fields_for_the_same_doc(self):
        doc1 = self._add_and_get_doc(self.text_index_1, "123", ["title"])
        self.assertEquals(1, len(doc1['_tensor_facets']))
        self.assertIn('title', doc1['_tensor_facets'][0])

        # override the same doc with different tensor_fields
        doc1 = self._add_and_get_doc(self.text_index_1, "123", ["desc"])
        self.assertEquals(1, len(doc1['_tensor_facets']))
        self.assertIn('desc', doc1['_tensor_facets'][0])

    def test_add_documents_should_use_existing_tensors_from_the_same_doc(self):
        doc1 = self._add_and_get_doc(self.text_index_1, "123", ["title"])

        with mock.patch('marqo.s2_inference.s2_inference.vectorise') as mock_vectorise:
            doc2 = self._add_and_get_doc(self.text_index_1, "123", ["title"],
                                         use_existing_tensors=True)
            self.assertFalse(mock_vectorise.called)
            self.assertEquals(doc1['_tensor_facets'], doc2['_tensor_facets'])

    def test_add_documents_should_add_string_fields_as_lexical_fields(self):
        self._add_and_get_doc(self.text_index_2, "123", [])

        updated_index = cast(SemiStructuredMarqoIndex, self.config.index_management.get_index(self.text_index_2))
        self.assertIn('title', updated_index.field_map)
        self.assertIn('desc', updated_index.field_map)
        self.assertIn('title', updated_index.lexically_searchable_fields_names)
        self.assertIn('desc', updated_index.lexically_searchable_fields_names)
        self.assertIn('marqo__lexical_title', updated_index.lexical_field_map)
        self.assertIn('marqo__lexical_desc', updated_index.lexical_field_map)

        res = tensor_search.search(
            text="content", search_method=SearchMethod.LEXICAL,
            config=self.config, index_name=self.text_index_2,
            searchable_attributes=['title']
        )
        self.assertEqual(1, len(res['hits']))

        res = tensor_search.search(
            text="content", search_method=SearchMethod.LEXICAL,
            config=self.config, index_name=self.text_index_2,
            searchable_attributes=['desc']
        )
        self.assertEqual(1, len(res['hits']))

    def test_add_documents_should_add_custom_vector_field_content_as_lexical_fields(self):
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.text_index_3,
                docs=[{
                    "title": "content 1",
                    "custom_vector_field": {
                        "content": "dog",
                        "vector": [0.2] * 32  # model has 32 dimensions
                    }
                }],
                device="cpu", tensor_fields=["custom_vector_field"],
                mappings={
                    "custom_vector_field": {"type": "custom_vector"}
                }
            )
        )
        res = tensor_search.search(
            text="dog", search_method=SearchMethod.LEXICAL,
            config=self.config, index_name=self.text_index_3,
            searchable_attributes=['custom_vector_field'],
            filter='custom_vector_field:dog'  # unstructured index uses short strings as filter, keep this behaviour
        )
        self.assertEqual(1, len(res['hits']))

        updated_index = cast(SemiStructuredMarqoIndex, self.config.index_management.get_index(self.text_index_3))
        self.assertIn('custom_vector_field', updated_index.field_map.keys())
        self.assertIn('marqo__lexical_custom_vector_field', updated_index.lexical_field_map.keys())

    def test_add_documents_should_add_image_field_as_lexical_fields(self):
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.image_index_with_chunking,
                docs=[{
                    "title": "content 1",
                    "image_field": TestImageUrls.HIPPO_REALISTIC.value
                }],
                device="cpu", tensor_fields=["image_field"]
            )
        )

        res = tensor_search.search(
            text="hippo", search_method=SearchMethod.LEXICAL,
            config=self.config, index_name=self.image_index_with_chunking,
            searchable_attributes=['image_field']
        )

        self.assertEqual(1, len(res['hits']))

        updated_index = cast(SemiStructuredMarqoIndex,
                             self.config.index_management.get_index(self.image_index_with_chunking))
        self.assertIn('image_field', updated_index.field_map.keys())
        self.assertIn('marqo__lexical_image_field', updated_index.lexical_field_map.keys())

    def test_add_documents_should_add_multimodal_subfield_as_lexical_fields(self):
        add_doc_result = self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.text_index_4,
                docs=[{
                    "title": "content 1",
                }],
                device="cpu", tensor_fields=["combo_field"],
                mappings={
                    "combo_field": {
                        "type": "multimodal_combination",
                        "weights": {"title": 1.0}
                    }
                }
            )
        )

        self.assertFalse(add_doc_result.errors)

        res = tensor_search.search(
            text="content", search_method=SearchMethod.LEXICAL,
            config=self.config, index_name=self.text_index_4,
            searchable_attributes=['title']
        )
        self.assertEqual(1, len(res['hits']))

        res = tensor_search.search(
            text="content", search_method=SearchMethod.TENSOR,
            config=self.config, index_name=self.text_index_4,
            searchable_attributes=['combo_field']
        )
        self.assertEqual(1, len(res['hits']))

        updated_index = cast(SemiStructuredMarqoIndex,
                             self.config.index_management.get_index(self.text_index_4))
        self.assertIn('title', updated_index.field_map.keys())
        self.assertIn('marqo__lexical_title', updated_index.lexical_field_map.keys())
        self.assertNotIn('combo_field', updated_index.field_map.keys())
        self.assertNotIn('marqo__lexical_combo_field', updated_index.lexical_field_map.keys())

    def test_add_documents_should_allow_the_same_field_to_have_different_types_in_different_batches(self):
        # batch 1: tensor field is a combo field
        self.config.document.add_documents(
            AddDocsParams(
                index_name=self.text_index_5,
                docs=[{
                    "_id": "1",
                    "title": "content 1",
                }],
                device="cpu", tensor_fields=["universal_tensor_field"],
                mappings={
                    "universal_tensor_field": {
                        "type": "multimodal_combination",
                        "weights": {"title": 1.0}
                    }
                }
            )
        )
        # batch 2: tensor field is a custom vector field
        self.config.document.add_documents(
            AddDocsParams(
                index_name=self.text_index_5,
                docs=[{
                    "_id": "2",
                    "title": "content 1",
                    "universal_tensor_field": {
                        "content": "content",
                        "vector": [0.2] * 32  # model has 32 dimensions
                    }
                }],
                device="cpu", tensor_fields=["universal_tensor_field"],
                mappings={
                    "universal_tensor_field": {"type": "custom_vector"}
                }
            )
        )
        # batch 3: tensor field is a text field
        self.config.document.add_documents(
            AddDocsParams(
                index_name=self.text_index_5,
                docs=[{
                    "_id": "3",
                    "universal_tensor_field": "content 1",
                }],
                device="cpu", tensor_fields=["universal_tensor_field"],
            )
        )
        # batch 4: same field name is used as a non-tensor field
        self.config.document.add_documents(
            AddDocsParams(
                index_name=self.text_index_5,
                docs=[{
                    "_id": "4",
                    "title": "content 1",
                    "universal_tensor_field": 1.0,  # used as a float field
                }],
                device="cpu", tensor_fields=["title"],
            )
        )

        res = tensor_search.search(
            text="content", search_method=SearchMethod.TENSOR,
            config=self.config, index_name=self.text_index_5,
            searchable_attributes=['universal_tensor_field']
        )
        self.assertEqual({'1', '2', '3'}, {hit['_id'] for hit in res['hits']})

        res = tensor_search.search(
            text="content", search_method=SearchMethod.LEXICAL,
            config=self.config, index_name=self.text_index_5,
            searchable_attributes=['universal_tensor_field']
        )
        # only the last 2 should return in a lexical search since nothing is indexed as universal_tensor_field
        # lexical field for doc 1
        self.assertEqual({'2', '3'}, {hit['_id'] for hit in res['hits']})

    def test_add_documents_should_raise_error_when_field_count_exceeds_limit(self):
        field_count_config = SemiStructuredFieldCountConfig(max_lexical_field_count=6, max_tensor_field_count=5)
        self.config.document.add_documents(
            AddDocsParams(
                index_name=self.text_index_6,
                docs=[{
                    "_id": "1",
                    "tensor_field1": "content 1",  # tensor fields are also lexical fields
                    "tensor_field2": "content 2",
                    "tensor_field3": "content 3",
                    "tensor_field4": "content 4",
                    "tensor_field5": "content 5",
                }],
                device="cpu", tensor_fields=[
                    "tensor_field1",
                    "tensor_field2",
                    "tensor_field3",
                    "tensor_field4",
                    "tensor_field5",
                ],
            ),
            field_count_config=field_count_config
        )

        with self.assertRaises(TooManyFieldsError) as err:
            self.config.document.add_documents(AddDocsParams(
                index_name=self.text_index_6,
                docs=[{
                    "_id": "2", "tensor_field6": "content 6"
                }],
                tensor_fields=["tensor_field6"]),
                field_count_config=field_count_config
            )
        self.assertIn('has 5 tensor fields. Your request to add tensor_field6 as a tensor field is '
                      'rejected since it exceeds the limit of 5. Please set a larger limit in '
                      'MARQO_MAX_TENSOR_FIELD_COUNT_UNSTRUCTURED environment variable.', str(err.exception))

        with self.assertRaises(TooManyFieldsError) as err2:
            self.config.document.add_documents(AddDocsParams(
                index_name=self.text_index_6,
                docs=[{
                    "_id": "3",
                    "tensor_field1": "content 1",
                    "lexical_field6": "content 6",
                    "lexical_field7": "content 7",
                }],
                tensor_fields=["tensor_field1"]),
                field_count_config=field_count_config
            )
        self.assertIn('has 6 lexical fields. Your request to add lexical_field7 as a lexical field is '
                      'rejected since it exceeds the limit of 6. Please set a larger limit in '
                      'MARQO_MAX_LEXICAL_FIELD_COUNT_UNSTRUCTURED environment variable.', str(err2.exception))

