import unittest
from typing import List, Optional

from pydantic.v1 import ValidationError

from marqo.core.models.marqo_index import CollapseField
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest


class TestUnstructuredMarqoIndexRequestCollapseFields(unittest.TestCase):
    """Unit tests for collapse_fields in UnstructuredMarqoIndexRequest."""

    def _create_base_unstructured_request(self, collapse_fields: Optional[List[CollapseField]] = None):
        """Helper to create a basic UnstructuredMarqoIndexRequest."""
        from marqo.core.models.marqo_index import (
            Model, TextPreProcessing, ImagePreProcessing, VideoPreProcessing, 
            AudioPreProcessing, DistanceMetric, VectorNumericType, HnswConfig, 
            TextSplitMethod, PatchMethod
        )
        
        return UnstructuredMarqoIndexRequest(
            name="test-index",
            model=Model(name="hf/all-MiniLM-L6-v2", text_query_prefix="", text_chunk_prefix=""),
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(patch_method=None),
            video_preprocessing=VideoPreProcessing(split_length=20, split_overlap=1),
            audio_preprocessing=AudioPreProcessing(split_length=20, split_overlap=1),
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=128, m=16),
            marqo_version="2.13.0",
            created_at=1234567890,
            updated_at=1234567890,
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=50,
            collapse_fields=collapse_fields
        )

    def test_unstructured_request_without_collapse_fields(self):
        """Test creating UnstructuredMarqoIndexRequest without collapse_fields."""
        request = self._create_base_unstructured_request()
        self.assertIsNone(request.collapse_fields)

    def test_unstructured_request_with_valid_collapse_fields(self):
        """Test creating UnstructuredMarqoIndexRequest with valid collapse_fields."""
        collapse_fields = [CollapseField(name="product_id", minGroups=100)]
        request = self._create_base_unstructured_request(collapse_fields=collapse_fields)
        
        self.assertEqual(len(request.collapse_fields), 1)
        self.assertEqual(request.collapse_fields[0].name, "product_id")
        self.assertEqual(request.collapse_fields[0].min_groups, 100)

    def test_unstructured_request_with_valid_collapse_field_name_and_min_groups(self):
        """Test with valid collapse field name and minGroups."""
        collapse_fields = [CollapseField(name="variant_group", minGroups=500)]
        request = self._create_base_unstructured_request(collapse_fields=collapse_fields)
        
        self.assertEqual(request.collapse_fields[0].name, "variant_group")
        self.assertEqual(request.collapse_fields[0].min_groups, 500)

    def test_unstructured_request_with_invalid_collapse_field_name(self):
        """Test with invalid collapse field name."""
        with self.assertRaises(ValidationError):
            CollapseField(name="marqo__reserved", minGroups=100)

    def test_unstructured_request_with_invalid_min_groups_negative(self):
        """Test with negative minGroups."""
        with self.assertRaises(ValidationError):
            CollapseField(name="product_id", minGroups=-1)

    def test_unstructured_request_with_invalid_min_groups_zero(self):
        """Test with zero minGroups."""
        with self.assertRaises(ValidationError):
            CollapseField(name="product_id", minGroups=0)

    def test_unstructured_request_with_multiple_collapse_fields_fails(self):
        """Test that multiple collapse fields are rejected."""
        collapse_fields = [
            CollapseField(name="product_id", minGroups=100),
            CollapseField(name="variant_id", minGroups=200)
        ]
        
        with self.assertRaises(ValidationError) as cm:
            self._create_base_unstructured_request(collapse_fields=collapse_fields)
        
        self.assertIn("Only one collapse field is supported", str(cm.exception))

    def test_unstructured_request_with_empty_collapse_fields_list_fails(self):
        """Test that empty collapse_fields list is rejected."""
        with self.assertRaises(ValidationError) as cm:
            self._create_base_unstructured_request(collapse_fields=[])
        
        self.assertIn("collapse_fields cannot be an empty list", str(cm.exception))


if __name__ == '__main__':
    unittest.main()