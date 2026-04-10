import unittest

from marqo.base_model import (
    MarqoBaseModelV2,
    StrictBaseModelV2,
    ImmutableBaseModelV2,
    ImmutableStrictBaseModelV2
)


class TestBaseModelV2Configs(unittest.TestCase):
    """Test V2 model_config dictionaries"""

    def test_marqo_base_model_v2_config(self):
        """Test MarqoBaseModelV2 model_config"""
        config = MarqoBaseModelV2.model_config
        self.assertEqual(config.get('validate_by_name'), True)
        self.assertEqual(config.get('validate_assignment'), True)

    def test_strict_base_model_v2_config(self):
        """Test StrictBaseModelV2 model_config"""
        config = StrictBaseModelV2.model_config
        self.assertEqual(config.get('validate_by_name'), True)
        self.assertEqual(config.get('validate_assignment'), True)
        self.assertEqual(config.get('extra'), 'forbid')

    def test_immutable_base_model_v2_config(self):
        """Test ImmutableBaseModelV2 model_config"""
        config = ImmutableBaseModelV2.model_config
        self.assertEqual(config.get('validate_by_name'), True)
        self.assertEqual(config.get('validate_assignment'), True)
        self.assertEqual(config.get('frozen'), True)

    def test_immutable_strict_base_model_v2_config(self):
        """Test ImmutableStrictBaseModelV2 model_config"""
        config = ImmutableStrictBaseModelV2.model_config
        self.assertEqual(config.get('validate_by_name'), True)
        self.assertEqual(config.get('validate_assignment'), True)
        self.assertEqual(config.get('frozen'), True)
        self.assertEqual(config.get('extra'), 'forbid')