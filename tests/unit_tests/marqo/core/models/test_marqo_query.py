import unittest

from marqo.core.models.marqo_query import MarqoTensorQuery


class TestMarqoTensorQuery(unittest.TestCase):
    def test_approximate_threshold_default_value(self):
        """Test that approximate_threshold is None by default"""
        query = MarqoTensorQuery(vector_query=[0.1, 0.2, 0.3])
        self.assertIsNone(query.approximate_threshold)
        self.assertTrue(query.approximate)  # Default should be True

    def test_approximate_threshold_setting(self):
        """Test that approximate_threshold can be set"""
        query = MarqoTensorQuery(vector_query=[0.1, 0.2, 0.3], approximate_threshold=0.5)
        self.assertEqual(query.approximate_threshold, 0.5)
        self.assertTrue(query.approximate)

    def test_approximate_threshold_with_approximate_false(self):
        """Test that approximate_threshold can be set even when approximate is False"""
        # This should be allowed at the model level, validation happens in API layer
        query = MarqoTensorQuery(vector_query=[0.1, 0.2, 0.3], approximate=False, approximate_threshold=0.5)
        self.assertEqual(query.approximate_threshold, 0.5)
        self.assertFalse(query.approximate)


if __name__ == "__main__":
    unittest.main()
