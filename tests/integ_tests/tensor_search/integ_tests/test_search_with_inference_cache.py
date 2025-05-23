from unittest.mock import patch

from integ_tests.marqo_test import MarqoTestCase
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model, UnstructuredMarqoIndex, FieldType, FieldFeature
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.inference.inference_cache.caching_inference import CachingInference
from marqo.tensor_search import tensor_search


class TestSearchWithInferenceCache(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.config.inference = CachingInference(delegate=cls.config.inference, cache_size=10, cache_type="LRU")

        # UNSTRUCTURED indexes
        unstructured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all_datasets_v4_MiniLM-L6')
        )

        # STRUCTURED indexes
        structured_default_text_index = cls.structured_marqo_index_request(
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter])
            ],

            tensor_fields=["text_field_1"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_default_text_index,
            structured_default_text_index,
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_default_text_index = cls.indexes[0]
        cls.structured_default_text_index = cls.indexes[1]

        # Add dummy document to all indexes in set up:
        for index in [cls.unstructured_default_text_index, cls.structured_default_text_index]:
            cls.add_documents(
                config=cls.config,
                add_docs_params=AddDocsParams(
                    index_name=index.name,
                    docs=[{"_id": "1", "text_field_1": "dummy"}],
                    tensor_fields=["text_field_1"] if isinstance(index, UnstructuredMarqoIndex) else None
                )
            )
            # populate the cache with key hello before the test
            tensor_search.search(
                text='hello', config=cls.config, index_name=index.name,
            )

    @patch('marqo.inference.native_inference.local_inference.NativeInferenceLocal.vectorise')
    def test_search_with_inference_cache(self, mock_vectorise):
        for index in [self.unstructured_default_text_index, self.structured_default_text_index]:
            with self.subTest(index=index.type):
                tensor_search.search(
                    text='hello', config=self.config, index_name=index.name,
                )

                # verify the vectorisation does not happen
                mock_vectorise.assert_not_called()
