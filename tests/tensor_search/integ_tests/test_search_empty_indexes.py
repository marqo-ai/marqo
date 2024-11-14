import os
from unittest import mock

from marqo.core.exceptions import InvalidFieldNameError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import HybridParameters
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from tests.marqo_test import MarqoTestCase


class TestSearchEmptyIndexes(MarqoTestCase):
    """
    Combined tests for search on indexes without tensor or lexical fields
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        unstructured_index_no_tensor_or_lexical_field = cls.unstructured_marqo_index_request(
            name='unstructured_index_no_tensor_or_lexical_field',
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            marqo_version='2.12.0'
        )

        semi_structured_index_no_tensor_or_lexical_field = cls.unstructured_marqo_index_request(
            name='semi_structured_index_no_tensor_or_lexical_field',
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
        )

        semi_structured_index_no_tensor_field = cls.unstructured_marqo_index_request(
            name='semi_structured_index_no_tensor_field',
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
        )

        structured_index_no_tensor_or_lexical_field = cls.structured_marqo_index_request(
            name='structured_index_no_tensor_or_lexical_field',
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="array_text_field", type=FieldType.ArrayText, features=[FieldFeature.Filter]),
                FieldRequest(name="int_field", type=FieldType.Int, features=[FieldFeature.Filter]),
            ],
            tensor_fields=[]
        )

        structured_index_no_tensor_field = cls.structured_marqo_index_request(
            name='structured_index_no_tensor_field',
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field", type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name="array_text_field", type=FieldType.ArrayText, features=[FieldFeature.Filter]),
                FieldRequest(name="int_field", type=FieldType.Int, features=[FieldFeature.Filter]),
            ],
            tensor_fields=[]
        )

        structured_index_no_lexical_field = cls.structured_marqo_index_request(
            name='structured_index_no_lexical_field',
            model=Model(name="hf/all_datasets_v4_MiniLM-L6"),
            fields=[
                FieldRequest(name="text_field", type=FieldType.Text, features=[FieldFeature.Filter]),
                FieldRequest(name="array_text_field", type=FieldType.ArrayText, features=[FieldFeature.Filter]),
                FieldRequest(name="int_field", type=FieldType.Int, features=[FieldFeature.Filter]),
            ],
            tensor_fields=["text_field"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_index_no_tensor_or_lexical_field,
            semi_structured_index_no_tensor_or_lexical_field,
            structured_index_no_tensor_or_lexical_field,
            structured_index_no_tensor_field,
            structured_index_no_lexical_field,
            semi_structured_index_no_tensor_field,
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_index_no_tensor_or_lexical_field = cls.indexes[0]
        cls.semi_structured_index_no_tensor_or_lexical_field = cls.indexes[1]
        cls.structured_index_no_tensor_or_lexical_field = cls.indexes[2]
        cls.structured_index_no_tensor_field = cls.indexes[3]
        cls.structured_index_no_lexical_field = cls.indexes[4]
        cls.semi_structured_index_no_tensor_field = cls.indexes[5]


        # create the lexical field in semi_structured_index
        cls.add_documents(config=cls.config, add_docs_params=AddDocsParams(
            index_name=cls.semi_structured_index_no_tensor_field.name,
            docs=[{"_id": "doc1", "text_field": "hello"}],
            tensor_fields=[],
        ))

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

        self.docs = [
            {"_id": "doc1", "array_text_field": ["a", "b"], "int_field": 1},
            {"_id": "doc2", "array_text_field": ["c", "d"], "int_field": 2},
        ]

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    # Test structured index behaviour
    def test_tensor_search_on_structured_index_without_tensor_field_should_raise_error(self):
        for index in [
            self.structured_index_no_tensor_or_lexical_field,
            self.structured_index_no_tensor_field,
        ]:

            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
            ))

            for test_case, query_string in [
                ('query term', 'a'),
                ('wildcard search', '*'),
                ('empty query term', '')
            ]:
                with self.subTest(msg=f'Tensor search query "{test_case}" on {index.name}'):
                    with self.assertRaisesStrict(InvalidArgumentError) as err:
                        tensor_search.search(config=self.config, index_name=index.name,
                                             text=query_string)

                    self.assertIn(' has no tensor fields, thus tensor search cannot be performed. '
                                  'Please create an index with a tensor field, or try a different search method.',
                                  str(err.exception))

    def test_tensor_search_on_structured_index_with_tensor_field_should_return_empty_result(self):
        for index in [
            self.structured_index_no_lexical_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs
            ))

            for test_case, query_string in [
                ('query term', 'a'),
                ('wildcard search', '*'),
                ('empty query term', '')
            ]:
                with self.subTest(msg=f'Tensor search query "{test_case}" on {index.name}'):
                    res = tensor_search.search(config=self.config, index_name=index.name, text=query_string)
                    self.assertEqual(0, len(res['hits']))

    def test_lexical_search_on_structured_index_without_lexical_field_should_raise_error(self):
        for index in [
            self.structured_index_no_tensor_or_lexical_field,
            self.structured_index_no_lexical_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
            ))

            for test_case, query_string in [
                ('query term', 'a'),
                ('wildcard search', '*'),
                ('empty query term', '')
            ]:
                with self.subTest(msg=f'Lexical search query "{test_case}" on {index.name}'):
                    with self.assertRaisesStrict(InvalidArgumentError) as err:
                        tensor_search.search(config=self.config, index_name=index.name,
                                             text=query_string, search_method=SearchMethod.LEXICAL)

                    self.assertIn(' has no lexically searchable fields, thus lexical search cannot be performed. '
                                  'Please create an index with a lexically searchable field, or try a different search method.',
                                  str(err.exception))

    def test_lexical_search_on_structured_index_with_lexical_field_should_return_expected_result(self):
        for index in [
            self.structured_index_no_tensor_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
            ))

            for test_case, query_string, expected_hits in [
                ('query term', 'a', 0),
                ('wildcard search', '*', 2),  # should still return all docs
                ('empty query term', '', 0)
            ]:
                with self.subTest(msg=f'Lexical search query "{test_case}" on {index.name}'):

                    res = tensor_search.search(config=self.config, index_name=index.name,
                                               text=query_string, search_method=SearchMethod.LEXICAL)

                    self.assertEqual(expected_hits, len(res['hits']))

    def test_hybrid_search_on_structured_index_without_tensor_or_lexical_field_should_raise_error(self):
        for index in [
            self.structured_index_no_tensor_or_lexical_field,
            self.structured_index_no_lexical_field,
            self.structured_index_no_tensor_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
            ))

            for test_case, query_string in [
                ('query term', 'a'),
                ('wildcard search', '*'),
                ('empty query term', '')
            ]:
                for retrival_method, ranking_method in [
                    ('disjunction', 'rrf'),
                    ('tensor', 'lexical'),
                    ('lexical', 'tensor'),
                ]:
                    with self.subTest(msg=f'Hybrid ({retrival_method} + {ranking_method}) search query "{test_case}" '
                                          f'on {index.name}'):
                        with self.assertRaisesStrict(InvalidArgumentError) as err:
                            tensor_search.search(config=self.config, index_name=index.name,
                                                 text=query_string, search_method=SearchMethod.HYBRID,
                                                 hybrid_parameters=HybridParameters(
                                                     retrievalMethod=retrival_method,
                                                     rankingMethod=ranking_method),
                                                 )

                        self.assertIn('either has no tensor fields or no lexically searchable fields, '
                                      'thus hybrid search cannot be performed. Please create an index with both tensor and '
                                      'lexical fields, or try a different search method.',
                                      str(err.exception))

    # Test unstructured (both legacy and new) index behaviour
    def test_tensor_search_on_unstructured_index_without_tensor_field_should_return_empty_result(self):
        for index in [
            self.semi_structured_index_no_tensor_or_lexical_field,
            self.semi_structured_index_no_tensor_field,
            self.unstructured_index_no_tensor_or_lexical_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
                tensor_fields=[]
            ))

            for test_case, query_string in [
                ('query term', 'a'),
                ('wildcard search', '*'),
                ('empty query term', '')
            ]:
                with self.subTest(msg=f'Tensor search query "{test_case}" on {index.name}'):
                    res = tensor_search.search(config=self.config, index_name=index.name, text=query_string)
                    self.assertEqual(0, len(res['hits']))

    def test_lexical_search_on_unstructured_index_should_return_expected_result(self):
        for index in [
            self.semi_structured_index_no_tensor_or_lexical_field,
            self.semi_structured_index_no_tensor_field,
            self.unstructured_index_no_tensor_or_lexical_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
                tensor_fields=[]
            ))

            for test_case, query_string, expected_hits in [
                ('query term', 'a', 0),
                ('wildcard search', '*', 2),  # should still return all docs
                ('empty query term', '', 0)
            ]:
                with self.subTest(msg=f'Lexical search query "{test_case}" on {index.name}'):

                    res = tensor_search.search(config=self.config, index_name=index.name,
                                               text=query_string, search_method=SearchMethod.LEXICAL)

                    self.assertEqual(expected_hits, len(res['hits']))

    def test_hybrid_search_on_unstructured_index_should_return_expected_result(self):
        for index in [
            self.semi_structured_index_no_tensor_or_lexical_field,
            self.semi_structured_index_no_tensor_field,
            self.unstructured_index_no_tensor_or_lexical_field
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
                tensor_fields=[]
            ))

            for test_case, query_string, retrival_method, ranking_method, expected_hits in [
                ('query term', 'a', 'disjunction', 'rrf', 0),
                ('wildcard search', '*', 'disjunction', 'rrf', 2),  # should still return all docs
                ('empty query term', '', 'disjunction', 'rrf', 0),

                ('query term', 'a', 'lexical', 'tensor', 0),
                ('wildcard search', '*', 'lexical', 'tensor', 2),  # should still return all docs
                ('empty query term', '', 'lexical', 'tensor', 0),

                ('query term', 'a', 'tensor', 'lexical', 0),
                ('wildcard search', '*', 'tensor', 'lexical', 0),  # tensor search won't return anything
                ('empty query term', '', 'tensor', 'lexical', 0),
            ]:
                with self.subTest(msg=f'Hybrid ({retrival_method} + {ranking_method}) search query "{test_case}" '
                                      f'on {index.name}'):

                    res = tensor_search.search(config=self.config, index_name=index.name,
                                               text=query_string, search_method=SearchMethod.HYBRID,
                                               hybrid_parameters=HybridParameters(
                                                   retrievalMethod=retrival_method,
                                                   rankingMethod=ranking_method),
                                               )

                    self.assertEqual(expected_hits, len(res['hits']))

    # Test searchable attributes
    def test_tensor_search_with_searchable_attributes_on_indexes_with_tensor_field_should_return_empty_result(self):
        for index in [
            self.structured_index_no_lexical_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
            ))

            for test_case, query_string in [
                ('query term', 'a'),
                ('wildcard search', '*'),
                ('empty query term', '')
            ]:
                with self.subTest(msg=f'Tensor search query "{test_case}" on {index.name}'):
                    res = tensor_search.search(config=self.config, index_name=index.name,
                                               text=query_string, searchable_attributes=['text_field'])
                    self.assertEqual(0, len(res['hits']))

    def test_tensor_search_with_searchable_attributes_on_indexes_without_tensor_field_should_raise_error(self):
        for index in [
            self.semi_structured_index_no_tensor_or_lexical_field,
            self.semi_structured_index_no_tensor_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
                tensor_fields=[]
            ))

            for test_case, query_string in [
                ('query term', 'a'),
                ('wildcard search', '*'),
                ('empty query term', '')
            ]:
                with self.subTest(msg=f'Tensor search query "{test_case}" on {index.name}'):
                    with self.assertRaisesStrict(InvalidFieldNameError) as err:
                        tensor_search.search(config=self.config, index_name=index.name,
                                             text=query_string, searchable_attributes=['text_field'])

                    self.assertIn('has no tensor field text_field.', str(err.exception))

    def test_lexical_search_with_searchable_attributes_on_indexes_with_lexical_field_should_return_expected_result(self):
        for index in [
            self.semi_structured_index_no_tensor_field,
            self.structured_index_no_tensor_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
                tensor_fields=[] if index.type == IndexType.SemiStructured else None
            ))

            for test_case, query_string, expected_hits in [
                ('query term', 'a', 0),
                ('wildcard search', '*', 2),  # should still return all docs
                ('empty query term', '', 0)
            ]:
                with self.subTest(msg=f'Tensor search query "{test_case}" on {index.name}'):
                    res = tensor_search.search(config=self.config, index_name=index.name,
                                               text=query_string, search_method=SearchMethod.LEXICAL,
                                               searchable_attributes=['text_field'])
                    self.assertEqual(expected_hits, len(res['hits']))

    def test_hybrid_search_with_lexical_searchable_attributes_on_indexes_should_return_expected_result(self):
        for index in [
            self.semi_structured_index_no_tensor_field,
        ]:
            self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=index.name,
                docs=self.docs,
                tensor_fields=[] if index.type == IndexType.SemiStructured else None
            ))

            for test_case, query_string, retrival_method, ranking_method, expected_hits in [
                ('query term', 'a', 'disjunction', 'rrf', 0),
                ('wildcard search', '*', 'disjunction', 'rrf', 2),
                ('empty query term', '', 'disjunction', 'rrf', 0),

                ('query term', 'a', 'lexical', 'tensor', 0),
                ('wildcard search', '*', 'lexical', 'tensor', 2),
                ('empty query term', '', 'lexical', 'tensor', 0),

                ('query term', 'a', 'tensor', 'lexical', 0),
                ('wildcard search', '*', 'tensor', 'lexical', 0),
                ('empty query term', '', 'tensor', 'lexical', 0),
            ]:
                with self.subTest(msg=f'Hybrid ({retrival_method} + {ranking_method}) search query "{test_case}" '
                                      f'on {index.name}'):

                    res = tensor_search.search(config=self.config, index_name=index.name,
                                               text=query_string, search_method=SearchMethod.HYBRID,
                                               hybrid_parameters=HybridParameters(
                                                   retrievalMethod=retrival_method,
                                                   rankingMethod=ranking_method,
                                                   searchableAttributesLexical=['text_field']
                                               ))

                    self.assertEqual(expected_hits, len(res['hits']))