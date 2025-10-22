import unittest

from inference_orchestrator import marqo_docs


class TestMarqoDocs(unittest.TestCase):
    """Tests for marqo_docs module."""

    def test_configuring_marqo(self):
        """Test configuring_marqo returns correct URL."""
        result = marqo_docs.configuring_marqo()
        self.assertIn("configuration", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_create_index(self):
        """Test create_index returns correct URL."""
        result = marqo_docs.create_index()
        self.assertIn("create-index", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_multimodal_combination_object(self):
        """Test multimodal_combination_object returns correct URL."""
        result = marqo_docs.multimodal_combination_object()
        self.assertIn("multimodal-combination-object", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_custom_vector_object(self):
        """Test custom_vector_object returns correct URL."""
        result = marqo_docs.custom_vector_object()
        self.assertIn("custom-vector-object", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_mappings(self):
        """Test mappings returns correct URL."""
        result = marqo_docs.mappings()
        self.assertIn("mappings", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_map_fields(self):
        """Test map_fields returns correct URL."""
        result = marqo_docs.map_fields()
        self.assertIn("map-fields", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_list_of_models(self):
        """Test list_of_models returns correct URL."""
        result = marqo_docs.list_of_models()
        self.assertIn("list-of-models", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_search_context(self):
        """Test search_context returns correct URL."""
        result = marqo_docs.search_context()
        self.assertIn("context", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_configuring_preloaded_models(self):
        """Test configuring_preloaded_models returns correct URL."""
        result = marqo_docs.configuring_preloaded_models()
        self.assertIn("preloaded-models", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_bring_your_own_model(self):
        """Test bring_your_own_model returns correct URL."""
        result = marqo_docs.bring_your_own_model()
        self.assertIn("bring-your-own-model", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_query_reference(self):
        """Test query_reference returns correct URL."""
        result = marqo_docs.query_reference()
        self.assertIn("query", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_indexing_images(self):
        """Test indexing_images returns correct URL."""
        result = marqo_docs.indexing_images()
        self.assertIn("images", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_api_reference_document_body(self):
        """Test api_reference_document_body returns correct URL."""
        result = marqo_docs.api_reference_document_body()
        self.assertIn("body", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_troubleshooting(self):
        """Test troubleshooting returns correct URL."""
        result = marqo_docs.troubleshooting()
        self.assertIn("troubleshooting", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_generic_models(self):
        """Test generic_models returns correct URL."""
        result = marqo_docs.generic_models()
        self.assertIn("generic-clip-models", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_search_api_score_modifiers_parameter(self):
        """Test search_api_score_modifiers_parameter returns correct URL."""
        result = marqo_docs.search_api_score_modifiers_parameter()
        self.assertIn("score-modifiers", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_hugging_face_trust_remote_code(self):
        """Test hugging_face_trust_remote_code returns correct URL."""
        result = marqo_docs.hugging_face_trust_remote_code()
        self.assertIn("hugging-face", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_update_documents_response(self):
        """Test update_documents_response returns correct URL."""
        result = marqo_docs.update_documents_response()
        self.assertIn("update-documents", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_hybrid_parameters(self):
        """Test hybrid_parameters returns correct URL."""
        result = marqo_docs.hybrid_parameters()
        self.assertIn("hybrid-parameters", result)
        self.assertIn(marqo_docs.base_url, result)

    def test_all_urls_contain_version(self):
        """Test that all generated URLs contain the docs version."""
        test_cases = [
            ("configuring_marqo", marqo_docs.configuring_marqo()),
            ("create_index", marqo_docs.create_index()),
            ("list_of_models", marqo_docs.list_of_models()),
        ]
        for msg, url in test_cases:
            with self.subTest(msg=msg):
                self.assertIn(marqo_docs.docs_version, url)


if __name__ == "__main__":
    unittest.main()
