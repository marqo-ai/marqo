import unittest

import pytest

from marqo.inference.native_inference.embedding_models.open_clip_model_properties import OpenCLIPModelProperties

@pytest.mark.unittest
class TestOpenCLIPModelProperties(unittest.TestCase):
    def test_both_original_and_alias_fields_work(self):
        """A test for creating a valid OpenCLIPModelProperties object with all fields with
        snake values and camel values."""
        test_cases = (
            (
                {
                    "name": "test-model",
                    "type": "open_clip",
                    "dimensions": 768,
                    "tokens": 256,
                    "modelLocation": {
                        "hf": {
                            "repoId": "test-repo-id",
                            "filename": "test-filename"
                        },
                    },
                    "imagePreprocessor": "SigLIP",
                },

                "alias fields/camelCase"),
            (
                {
                    "name": "test-model",
                    "type": "open_clip",
                    "dimensions": 768,
                    "tokens": 256,
                    "model_location": {
                        "hf": {
                            "repo_id": "test-repo-id",
                            "filename": "test-filename"
                        },
                    },
                    "image_preprocessor": "SigLIP",

                },
                "original fields/snake_case"),
        )

        for model_properties, msg in test_cases:
            with self.subTest(msg):
                open_clip_model_properties = OpenCLIPModelProperties(**model_properties)
                self.assertEqual(open_clip_model_properties.name, "test-model")
                self.assertEqual(open_clip_model_properties.type, "open_clip")
                self.assertEqual(open_clip_model_properties.dimensions, 768)
                self.assertEqual(open_clip_model_properties.jit, False)
                self.assertEqual(open_clip_model_properties.precision, "fp32")
                self.assertIsNone(open_clip_model_properties.url)
                self.assertIsNone(open_clip_model_properties.localpath)
                self.assertEqual(open_clip_model_properties.model_location.hf.repo_id, "test-repo-id")
                self.assertEqual(open_clip_model_properties.model_location.hf.filename, "test-filename")
                self.assertEqual(open_clip_model_properties.tokenizer, None)
                self.assertEqual(open_clip_model_properties.image_preprocessor, "SigLIP")
                self.assertEqual(open_clip_model_properties.mean, None)
                self.assertEqual(open_clip_model_properties.std, None)
                self.assertEqual(open_clip_model_properties.size, None)
                self.assertEqual(open_clip_model_properties.note, None)
                self.assertEqual(open_clip_model_properties.pretrained, None)