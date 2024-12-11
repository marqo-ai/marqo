import unittest

from marqo.s2_inference.model_registry import _get_open_clip_properties, load_model_properties
from tests.marqo_test import MarqoTestCase


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.open_clip_properties = _get_open_clip_properties()

    def test_open_clip_model_configurations_match(self):
        with self.subTest("Check if Marqo/ViT-B-32.openai & open_clip/ViT-B-32/openai configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-32.openai"],self.open_clip_properties["open_clip/ViT-B-32/openai"])

        with self.subTest("Check if Marqo/ViT-B-32.laion400m_e31 & open_clip/ViT-B-32/laion400m_e31 configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-32.laion400m_e31"],self.open_clip_properties["open_clip/ViT-B-32/laion400m_e31"])


        with self.subTest("Check if Marqo/ViT-B-32.laion400m_e32 & open_clip/ViT-B-32/laion400m_e32 configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-32.laion400m_e32"],self.open_clip_properties["open_clip/ViT-B-32/laion400m_e32"])

        with self.subTest("Check if Marqo/ViT-B-32.laion2b_e16 & open_clip/ViT-B-32/laion2b_e16 configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-32.laion2b_e16"],self.open_clip_properties["open_clip/ViT-B-32/laion2b_e16"])

        with self.subTest("Check if Marqo/ViT-B-32.laion2b_s34b_b79k & open_clip/ViT-B-32/laion2b_s34b_b79k configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-32.laion2b_s34b_b79k"],self.open_clip_properties["open_clip/ViT-B-32/laion2b_s34b_b79k"])

        with self.subTest("Check if Marqo/ViT-B-16.openai & open_clip/ViT-B-16/openai configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-16.openai"],self.open_clip_properties["open_clip/ViT-B-16/openai"])

        with self.subTest("Check if Marqo/ViT-B-16.laion400m_e31 & open_clip/ViT-B-16/laion400m_e31 configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-16.laion400m_e31"],self.open_clip_properties["open_clip/ViT-B-16/laion400m_e31"])

        with self.subTest("Check if Marqo/ViT-B-16.laion400m_e32 & open_clip/ViT-B-16/laion400m_e32 configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-16.laion400m_e32"],self.open_clip_properties["open_clip/ViT-B-16/laion400m_e32"])

        with self.subTest("Check if Marqo/ViT-B-16.laion2b_s34b_b88k & open_clip/ViT-B-16/laion2b_s34b_b88k configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-B-16.laion2b_s34b_b88k"],self.open_clip_properties["open_clip/ViT-B-16/laion2b_s34b_b88k"])

        with self.subTest("Check if Marqo/ViT-L-14.openai & open_clip/ViT-L-14/openai configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-L-14.openai"],self.open_clip_properties["open_clip/ViT-L-14/openai"])

        with self.subTest("Check if Marqo/ViT-L-14.laion400m_e31 & open_clip/ViT-L-14/laion400m_e31 configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-L-14.laion400m_e31"],self.open_clip_properties["open_clip/ViT-L-14/laion400m_e31"])

        with self.subTest("Check if Marqo/ViT-L-14.laion400m_e32 & open_clip/ViT-L-14/laion400m_e32 configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-L-14.laion400m_e32"],self.open_clip_properties["open_clip/ViT-L-14/laion400m_e32"])

        with self.subTest("Check if Marqo/ViT-L-14.laion2b_s32b_b82k & open_clip/ViT-L-14/laion2b_s32b_b82k configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/ViT-L-14.laion2b_s32b_b82k"],self.open_clip_properties["open_clip/ViT-L-14/laion2b_s32b_b82k"])

        with self.subTest("Check if Marqo/roberta-ViT-B-32.laion2b_s12b_b32k & open_clip/roberta-ViT-B-32/laion2b_s12b_b32k configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/roberta-ViT-B-32.laion2b_s12b_b32k"],self.open_clip_properties["open_clip/roberta-ViT-B-32/laion2b_s12b_b32k"])

        with self.subTest("Check if Marqo/xlm-roberta-base-ViT-B-32.laion5b_s13b_b90k & open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/xlm-roberta-base-ViT-B-32.laion5b_s13b_b90k"],self.open_clip_properties["open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k"])

        with self.subTest("Check if Marqo/xlm-roberta-large-ViT-H-14.frozen_laion5b_s13b_b90k & open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k configurations match"):
            self.compare_models(self.open_clip_properties["Marqo/xlm-roberta-large-ViT-H-14.frozen_laion5b_s13b_b90k"],self.open_clip_properties["open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k"])

    def compare_models(self, model_properties_with_name_in_marqtune_convention, model_properties):
        self.assertEqual(model_properties_with_name_in_marqtune_convention["dimensions"], model_properties["dimensions"], f"dimensions do not match for {model_properties_with_name_in_marqtune_convention['name']} and {model_properties['name']}")
        self.assertEqual(model_properties_with_name_in_marqtune_convention["note"], model_properties["note"], f"note do not match for {model_properties_with_name_in_marqtune_convention['name']} and {model_properties['name']}")
        self.assertEqual(model_properties_with_name_in_marqtune_convention["type"], model_properties["type"], f"type do not match for {model_properties_with_name_in_marqtune_convention['name']} and {model_properties['name']}")
        self.assertEqual(model_properties_with_name_in_marqtune_convention["pretrained"], model_properties["pretrained"], f"pretrained do not match for {model_properties_with_name_in_marqtune_convention['name']} and {model_properties['name']}")

