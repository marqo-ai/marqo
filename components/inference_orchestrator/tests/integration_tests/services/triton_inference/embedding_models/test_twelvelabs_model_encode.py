import os
import unittest

import numpy as np

from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.triton_inference.embedding_models.twelvelabs.twelvelabs_model import (
    TwelveLabsModel,
)


@unittest.skipUnless(
    os.environ.get("TWELVELABS_API_KEY"),
    "TWELVELABS_API_KEY is not set; skipping TwelveLabs Marengo API test.",
)
class TestTwelveLabsModelEncode(unittest.TestCase):
    """End-to-end test against the live TwelveLabs Marengo API.

    Skipped unless TWELVELABS_API_KEY is set. Get a free key with a generous
    free tier at https://twelvelabs.io .
    """

    def setUp(self):
        self.model = TwelveLabsModel(
            {
                "type": "twelvelabs",
                "name": "Marqo/marengo-3.0",
                "dimensions": 512,
                "apiModelName": "marengo3.0",
            }
        )
        self.model.load()

    def test_text_embedding_is_512_dim_and_normalised(self):
        embeddings = self.model.encode(
            inputs=["a red sports car"], modality=Modality.TEXT, normalize=True
        )
        self.assertEqual(1, len(embeddings))
        self.assertEqual((512,), embeddings[0].shape)
        self.assertAlmostEqual(float(np.linalg.norm(embeddings[0])), 1.0, places=4)

    def test_text_embeddings_are_deterministic(self):
        a = self.model.encode(["a cat"], modality=Modality.TEXT, normalize=True)[0]
        b = self.model.encode(["a cat"], modality=Modality.TEXT, normalize=True)[0]
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
