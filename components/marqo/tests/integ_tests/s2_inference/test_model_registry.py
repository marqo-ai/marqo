import unittest

from marqo.s2_inference.model_registry import _get_open_clip_properties, load_model_properties

class TestModelRegistry(unittest.TestCase):
    """
    Test case for verifying the configurations of different models in the model registry.
    """
    def setUp(self):
        self.open_clip_properties = _get_open_clip_properties()

    def test_open_clip_model_configurations_match(self):
        """
        Test to verify that the configurations of Marqo (which are following Marqtune convention) and open_clip models match.

        This test iterates over a list of model pairs and compares their configurations
        to ensure they are identical.

        The model pairs are defined as tuples of Marqo model names and their corresponding
        open_clip model names.

        Raises:
            AssertionError: If the configurations of any model pair do not match.
        """
        model_pairs = [
            ("Marqo/ViT-B-32.laion400m_e31", "open_clip/ViT-B-32/laion400m_e31"),
            ("Marqo/ViT-B-32.laion400m_e32", "open_clip/ViT-B-32/laion400m_e32"),
            ("Marqo/ViT-B-32.laion2b_e16", "open_clip/ViT-B-32/laion2b_e16"),
            ("Marqo/ViT-B-32.laion2b_s34b_b79k", "open_clip/ViT-B-32/laion2b_s34b_b79k"),
            ("Marqo/ViT-B-16.openai", "open_clip/ViT-B-16/openai"),
            ("Marqo/ViT-B-16.laion400m_e31", "open_clip/ViT-B-16/laion400m_e31"),
            ("Marqo/ViT-B-16.laion400m_e32", "open_clip/ViT-B-16/laion400m_e32"),
            ("Marqo/ViT-B-16.laion2b_s34b_b88k", "open_clip/ViT-B-16/laion2b_s34b_b88k"),
            ("Marqo/ViT-L-14.openai", "open_clip/ViT-L-14/openai"),
            ("Marqo/ViT-L-14.laion400m_e31", "open_clip/ViT-L-14/laion400m_e31"),
            ("Marqo/ViT-L-14.laion400m_e32", "open_clip/ViT-L-14/laion400m_e32"),
            ("Marqo/ViT-L-14.laion2b_s32b_b82k", "open_clip/ViT-L-14/laion2b_s32b_b82k"),
            ("Marqo/roberta-ViT-B-32.laion2b_s12b_b32k", "open_clip/roberta-ViT-B-32/laion2b_s12b_b32k"),
            ("Marqo/xlm-roberta-base-ViT-B-32.laion5b_s13b_b90k",
             "open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k"),
            ("Marqo/xlm-roberta-large-ViT-H-14.frozen_laion5b_s13b_b90k",
             "open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k")
        ]

        for marqo_model, open_clip_model in model_pairs:
            with self.subTest(f"Check if {marqo_model} & {open_clip_model} configurations match"):
                self.compare_models(self.open_clip_properties[marqo_model], self.open_clip_properties[open_clip_model])

    def compare_models(self, model_properties_with_name_in_marqtune_convention, model_properties):
        self.assertEqual(model_properties_with_name_in_marqtune_convention["dimensions"], model_properties["dimensions"], f"dimensions do not match for {model_properties_with_name_in_marqtune_convention['name']} and {model_properties['name']}")
        self.assertEqual(model_properties_with_name_in_marqtune_convention["note"], model_properties["note"], f"note do not match for {model_properties_with_name_in_marqtune_convention['name']} and {model_properties['name']}")
        self.assertEqual(model_properties_with_name_in_marqtune_convention["type"], model_properties["type"], f"type do not match for {model_properties_with_name_in_marqtune_convention['name']} and {model_properties['name']}")
        self.assertEqual(model_properties_with_name_in_marqtune_convention["pretrained"], model_properties["pretrained"], f"pretrained do not match for {model_properties_with_name_in_marqtune_convention['name']} and {model_properties['name']}")

