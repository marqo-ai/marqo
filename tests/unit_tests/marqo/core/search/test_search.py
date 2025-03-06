from marqo.tensor_search import tensor_search
from tests.unit_tests.marqo.base_test_case import BaseUnitTest


class SearchTest(BaseUnitTest):
    def test_tensor_search(self):
        tensor_search.search(self.config, "index_name", "query", search_method="tensor")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], self.get_expected_tensor_yql())
        self.assertEqual(call_args['model_restrict'], 'test_schema')
        self.assertEqual(call_args['hits'], 3)
        self.assertEqual(call_args['offset'], 0)

    def test_tensor_search_with_target_hits(self):
        tensor_search.search(self.config, "index_name", "query", search_method="tensor", target_hits=5)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], self.get_expected_tensor_yql(target_hits=5))

    def test_lexical_search(self):
        tensor_search.search(self.config, "index_name", "query", search_method="lexical")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], self.get_expected_lexical_yql("query"))
        self.assertEqual(call_args['query_features'], {'text_field_2': 1, 'text_field_1': 1})
        self.assertEqual(call_args['ranking'], 'bm25')
        self.assertEqual(call_args['hits'], 3)
        self.assertEqual(call_args['offset'], 0)
        self.assertEqual(call_args['model_restrict'], 'test_schema')
        self.assertEqual(call_args['presentation.summary'], 'all-non-vector-summary')

    def test_hybrid_search(self):
        tensor_search.search(self.config, "index_name", "query", search_method="hybrid")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.tensor'],
            self.get_expected_tensor_yql()
        )
        self.assertEqual(
            call_args['marqo__yql.lexical'], self.get_expected_lexical_yql("query")
        )

    def test_hybrid_search_with_target_hits(self):
        tensor_search.search(self.config, "index_name", "query", search_method="hybrid", target_hits=5)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.tensor'],
            self.get_expected_tensor_yql(target_hits=5)
        )
        self.assertEqual(
            call_args['marqo__yql.lexical'], self.get_expected_lexical_yql("query")
        )