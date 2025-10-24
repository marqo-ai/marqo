import unittest
import time
from typing import List, Optional

import marqo.api.exceptions as api_exceptions
import marqo.core.models.marqo_index as core
from marqo.core.models.marqo_index import CollapseField, SemiStructuredMarqoIndex
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.tensor_search.models.index_settings import IndexSettings


class TestIndexSettingsCollapseFields(unittest.TestCase):
    """Unit tests for collapseFields in IndexSettings."""

    def _create_base_index_settings(self, collapse_fields: Optional[List[CollapseField]] = None, 
                                   index_type: core.IndexType = core.IndexType.SemiStructured):
        """Helper to create a basic IndexSettings."""
        return IndexSettings(
            type=index_type,
            collapseFields=collapse_fields,
            model="hf/all-MiniLM-L6-v2",
            normalizeEmbeddings=True
        )

    def test_index_settings_without_collapse_fields(self):
        """Test creating IndexSettings without collapseFields."""
        settings = self._create_base_index_settings()
        self.assertIsNone(settings.collapseFields)

    def test_index_settings_with_valid_collapse_fields(self):
        """Test creating IndexSettings with valid collapseFields."""
        collapse_fields = [CollapseField(name="product_id", minGroups=100)]
        settings = self._create_base_index_settings(collapse_fields=collapse_fields)
        
        self.assertEqual(len(settings.collapseFields), 1)
        self.assertEqual(settings.collapseFields[0].name, "product_id")
        self.assertEqual(settings.collapseFields[0].min_groups, 100)

    def test_index_settings_collapse_fields_structured_index_fails(self):
        """Test that collapseFields with Structured index type fails."""
        collapse_fields = [CollapseField(name="product_id", minGroups=100)]
        
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            self._create_base_index_settings(
                collapse_fields=collapse_fields,
                index_type=core.IndexType.Structured
            )
        
        self.assertIn("collapseFields is only supported for unstructured indexes", str(cm.exception))

    def test_to_marqo_index_request_conversion(self):
        """Test that collapseFields correctly converts to collapse_fields in UnstructuredMarqoIndexRequest."""
        collapse_fields = [CollapseField(name="variant_group", minGroups=500)]
        settings = self._create_base_index_settings(collapse_fields=collapse_fields)
        
        request = settings.to_marqo_index_request("test-index")
        
        self.assertIsInstance(request, UnstructuredMarqoIndexRequest)
        self.assertEqual(len(request.collapse_fields), 1)
        self.assertEqual(request.collapse_fields[0].name, "variant_group")
        self.assertEqual(request.collapse_fields[0].min_groups, 500)

    def test_to_marqo_index_request_none_collapse_fields(self):
        """Test conversion when collapseFields is None."""
        settings = self._create_base_index_settings()
        
        request = settings.to_marqo_index_request("test-index")
        
        self.assertIsInstance(request, UnstructuredMarqoIndexRequest)
        self.assertIsNone(request.collapse_fields)

    def test_from_marqo_index_semi_structured_with_collapse_fields(self):
        """Test creating IndexSettings from SemiStructuredMarqoIndex with collapse_fields."""
        # Create a mock SemiStructuredMarqoIndex
        collapse_fields = [CollapseField(name="product_id", minGroups=200)]
        
        # Create a minimal SemiStructuredMarqoIndex for testing
        marqo_index = SemiStructuredMarqoIndex(
            name="test-index",
            model=core.Model(name="hf/all-MiniLM-L6-v2", text_query_prefix="", text_chunk_prefix=""),
            normalize_embeddings=True,
            text_preprocessing=core.TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=core.TextSplitMethod.Sentence
            ),
            image_preprocessing=core.ImagePreProcessing(patch_method=None),
            video_preprocessing=core.VideoPreProcessing(split_length=20, split_overlap=1),
            audio_preprocessing=core.AudioPreProcessing(split_length=20, split_overlap=1),
            distance_metric=core.DistanceMetric.Angular,
            vector_numeric_type=core.VectorNumericType.Float,
            hnsw_config=core.HnswConfig(ef_construction=128, m=16),
            marqo_version="2.23.0",
            created_at=int(time.time()),
            updated_at=int(time.time()),
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=50,
            schema_name="test_schema",
            lexical_fields=[],
            tensor_fields=[],
            collapse_fields=collapse_fields
        )
        
        settings = IndexSettings.from_marqo_index(marqo_index)
        
        self.assertEqual(len(settings.collapseFields), 1)
        self.assertEqual(settings.collapseFields[0].name, "product_id")
        self.assertEqual(settings.collapseFields[0].min_groups, 200)


if __name__ == '__main__':
    unittest.main()