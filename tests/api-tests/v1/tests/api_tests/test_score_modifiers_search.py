import uuid

import numpy as np
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestScoreModifierSearch(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.unstructured_score_modifier_index_name = ("unstructured_score_modifier"
                                                      + str(uuid.uuid4()).replace('-', ''))
        cls.structured_score_modifier_index_name = "structured_score_modifier" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.unstructured_score_modifier_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/laion400m_e31"
            },
            {
                "indexName": cls.structured_score_modifier_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/laion400m_e31",
                "allFields": [
                    {"name": "text_field", "type": "text"},
                    {"name": "image_field", "type": "image_pointer"},
                    {"name": "multiply_1", "type": "float", "features": ["score_modifier"]},
                    {"name": "multiply_2", "type": "float", "features": ["score_modifier"]},
                    {"name": "add_1", "type": "float", "features": ["score_modifier"]},
                    {"name": "add_2", "type": "float", "features": ["score_modifier"]},
                ],
                "tensorFields": ["text_field", "image_field"]
            }
        ]
        )

        cls.indexes_to_delete = [cls.structured_score_modifier_index_name,
                                 cls.unstructured_score_modifier_index_name]

    def test_score_modifier_search_results(self):
        for index_name in [self.unstructured_score_modifier_index_name, self.structured_score_modifier_index_name]:
            for _ in range(10):
                # Generate 8 random values to test score modifiers
                multiply_1_value, multiply_1_weight, multiply_2_value, multiply_2_weight, \
                    add_1_value, add_1_weight, add_2_value, add_2_weight = \
                    np.round(np.random.uniform(-10, 10, 8), 2)

                doc = {
                    "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue_small.png",
                    "text_field": "Marqo can support vector search",
                    "multiply_1": multiply_1_value,
                    "multiply_2": multiply_2_value,
                    "add_1": add_1_value,
                    "add_2": add_2_value,
                }

                score_modifiers = {
                    "multiply_score_by":
                        [{"field_name": "multiply_1", "weight": multiply_1_weight},
                         {"field_name": "multiply_2", "weight": multiply_2_weight}],
                    "add_to_score":
                        [{"field_name": "add_1", "weight": add_1_weight},
                         {"field_name": "add_2", "weight": add_2_weight}]
                }
                msg = (f"{'unstructured' if index_name.startswith('un') else 'structured'}, doc = {doc}, "
                       f"score_modifiers = {score_modifiers}")

                with self.subTest(msg):
                    self.clear_indexes(self.indexes_to_delete)
                    res = self.client.index(index_name).add_documents(
                        documents=[doc], tensor_fields=["text_field", "image_field"] if msg.startswith("un") else None,
                    )
                    self.assertEqual(1, self.client.index(index_name).get_stats()["numberOfDocuments"])

                    original_res = self.client.index(index_name).search(q="test_search", score_modifiers=None)
                    modifiers_res = self.client.index(index_name).search(q="test_search",
                                                                         score_modifiers=score_modifiers)
                    original_score = original_res["hits"][0]["_score"]
                    modifiers_score = modifiers_res["hits"][0]["_score"]

                    expected_score = original_score * multiply_1_value * multiply_1_weight * multiply_2_value \
                                     * multiply_2_weight + add_1_value * add_1_weight + add_2_value * add_2_weight

                    self.assertAlmostEqual(expected_score, modifiers_score, delta=1e-3)

    def test_invalid_score_modifiers_format(self):
        invalid_score_modifiers = {
            # typo in multiply score by
            "multiply_score_bys":
                [{"field_name": "multiply_1",
                  "weight": 1, },
                 {"field_name": "multiply_2", }],
            "add_to_score": [
                {"field_name": "add_1", "weight": 4,
                 },
                {"field_name": "add_2", "weight": 1,
                 }]
        }

        for index_name in [self.unstructured_score_modifier_index_name, self.structured_score_modifier_index_name]:
            with self.subTest(index_name):
                with self.assertRaises(MarqoWebError) as e:
                    res = self.client.index(index_name).search(
                        "query", score_modifiers=invalid_score_modifiers
                    )

                self.assertIn("score_modifiers", str(e.exception.message))


    def test_valid_score_modifiers_format(self):
        valid_score_modifiers_list = [
            {
                # missing add to score
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # missing multiply score by
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1, },
                     {"field_name": "multiply_2"}],
            }]

        for index_name in [self.unstructured_score_modifier_index_name, self.structured_score_modifier_index_name]:
            for valid_score_modifiers in valid_score_modifiers_list:
                with self.subTest(f"{index_name} - {valid_score_modifiers}"):
                    self.client.index(index_name).search("test", score_modifiers=valid_score_modifiers)