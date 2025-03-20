import unittest

from marqo.inference.native_inference.embedding_models.hugging_face_stella_model import HuggingFaceStellaModel
from marqo.s2_inference.errors import InvalidModelPropertiesError


class TestHuggingFaceStellaModel(unittest.TestCase):
    def test_trust_remote_code_validation(self):
        """
        Test that an error is raised when trust_remote_code is set to Flase or not set in model_properties.
        """
        for trust_remote_code in [None, False]:
            with self.subTest(trust_remote_code=trust_remote_code):
                model_properties = {k: v for k, v in {'name': 'my_model', 'type': 'hf', 'dimensions': 512,
                                                      'trustRemoteCode': trust_remote_code}.items() if
                                    v is not None}
                device = 'cpu'

                with self.assertRaises(InvalidModelPropertiesError) as context:
                    HuggingFaceStellaModel(model_properties, device)

                self.assertIn("trustRemoteCode", str(context.exception))

        with self.subTest(trust_remote_code=True):
            model_properties = {'name': 'my_model', 'type': 'hf', 'dimensions': 512,
                                'trustRemoteCode': True}
            device = 'cpu'

            self.assertIsNotNone(HuggingFaceStellaModel(model_properties, device))
