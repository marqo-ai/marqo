import traceback

import pytest

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase

@pytest.mark.marqo_version('2.12.0')
class TestCreateIndexBringYourOwnModel(BaseCompatibilityTestCase):
    load_from_hf_index_name = "test_create_index_api_bring_your_own_model"

    load_from_hf_index_settings = {
        "treatUrlsAndPointersAsImages": True,
        "model": "marqo-fashion-clip-custom-load",
        "modelProperties": {
            "name": "hf-hub:Marqo/marqo-fashionCLIP",
            "dimensions": 512,
            "type": "open_clip",
        },
        "normalizeEmbeddings": True,
    }
    load_from_public_url_index_name = "test_create_index_api_public_url"
    load_from_public_url_settings = {
        "treatUrlsAndPointersAsImages": True,
        "model": "my-own-clip-model",
        "modelProperties": {
            "name": "ViT-B-32",
            "dimensions": 512,
            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
            "type": "open_clip",
        },
        "normalizeEmbeddings": True,
    }
    load_from_public_url_with_custom_configurations_index_name = "test_create_index_api_public_url_custom_configurations"
    load_from_public_url_with_custom_configurations = {
        "treatUrlsAndPointersAsImages": True,
        "model": "my-own-clip-model",
        "modelProperties": {
            "name": "ViT-B-16-SigLIP",
            "dimensions": 768,
            "url": "https://huggingface.co/Marqo/marqo-fashionSigLIP/resolve/main/open_clip_pytorch_model.bin",
            "imagePreprocessor": "SigLIP",
            "type": "open_clip",
        },
        "normalizeEmbeddings": True,
    }
    indexes_to_test_on = [load_from_hf_index_name, load_from_public_url_index_name, load_from_public_url_with_custom_configurations_index_name]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.indexes_to_delete = cls.indexes_to_test_on
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        cls.indexes_to_delete = cls.indexes_to_test_on
        super().setUpClass()

    def prepare(self):

        all_results = {}
        errors = [] # To store errors in case of failure
        for index_name, settings in [(self.load_from_hf_index_name, self.load_from_hf_index_settings), (self.load_from_public_url_index_name, self.load_from_public_url_settings), (self.load_from_public_url_with_custom_configurations_index_name, self.load_from_public_url_with_custom_configurations)]:
            try:
                self.logger.debug(f"Creating index {index_name}")
                self.client.create_index(index_name = index_name, settings_dict = settings)
                all_results[index_name] = self.client.index(index_name).get_settings()
            except Exception as e:
                errors.append((index_name, traceback.format_exc()))

        if errors:
            failure_message = "\n".join([
                f"Failure in index {idx}, {error}"
                for idx, error in errors
            ])
            self.logger.error(f"Some subtests failed:\n{failure_message}. When the corresponding test runs for this index, it is expected to fail")

        self.save_results_to_file(all_results)

    def test_expected_settings(self):
        expected_settings = self.load_results_from_file()
        test_failures = [] # To store test_failures in case of failure

        for index_name in [self.load_from_hf_index_name, self.load_from_public_url_index_name, self.load_from_public_url_with_custom_configurations_index_name]:
            try:
                expected_setting = expected_settings[index_name]
                actual_setting = self.client.index(index_name).get_settings()
                self.assertEqual(expected_setting, actual_setting, f"Index settings do not match expected settings, expected {expected_setting}, but got {actual_setting}")
            except Exception as e:
                test_failures.append((index_name, traceback.format_exc()))

        if test_failures:
            failure_message = "\n".join([
                f"Failure in index {idx}: {error}"
                for idx, error in test_failures
            ])
            self.fail(f"Some subtests failed:\n{failure_message}")
