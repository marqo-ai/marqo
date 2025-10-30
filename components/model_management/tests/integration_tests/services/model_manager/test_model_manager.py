"""Integration tests for ModelManager with real model properties."""

import os
import shutil
import tempfile
import threading
from unittest import TestCase
from unittest.mock import MagicMock, patch

import requests

from model_management.config import get_config
from model_management.schemas.triton_model_properties import TritonModelProperties
from model_management.services.errors import ModelOperationInProgressError
from model_management.services.model_manager.model_manager import ModelManager
from model_management.services.triton.triton_client import TritonClient
from .expected_model_config import model_config

# Real model properties from actual Marqo models
test_model_properties = {
    "marqo-fashionSigLIP-image-encoder": {
        "maxBatchSize": 8,
        "name": "marqo-fashionSigLIP-image-encoder",
        "sources": [
            "s3://marqo-opensource-models/marqo-fashionSigLIP/image-encoder/model.onnx",
        ],
        "input": [{"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}],
        "output": [{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
    },
    "all-MiniLM-L6-v2-text-encoder": {
        "maxBatchSize": 16,
        "name": "all-MiniLM-L6-v2-text-encoder",
        "sources": [
            "s3://marqo-opensource-models/sentence-transformers-all-minilm-l6-v2/model.onnx"
        ],
        "input": [
            {"name": "input_ids", "dims": [-1], "dataType": "TYPE_INT64"},
            {"name": "attention_mask", "dims": [-1], "dataType": "TYPE_INT64"},
            {"name": "token_type_ids", "dims": [-1], "dataType": "TYPE_INT64"},
        ],
        "output": [
            {
                "name": "last_hidden_state",
                "dims": [-1, 384],
                "dataType": "TYPE_FP32",
            }
        ],
    },
}


class TestModelManager(TestCase):
    """Integration tests for ModelManager with realistic model configurations."""

    def setUp(self):
        """Set up test fixtures for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_triton_client = MagicMock(spec=TritonClient)
        self.model_manager = ModelManager(
            marqo_model_cache_path=self.temp_dir, triton_client=self.mock_triton_client
        )

    def tearDown(self):
        """Clean up test fixtures after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _get_model_props(self, model_key):
        """Helper to get TritonModelProperties for a test model."""
        return TritonModelProperties(**test_model_properties[model_key])

    def _create_model_files(self, model_name, include_version_dirs=False):
        """Helper to create fake model files in temp directory."""
        model_dir = os.path.join(self.temp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        if include_version_dirs:
            version_dir = os.path.join(model_dir, "1")
            os.makedirs(version_dir, exist_ok=True)
            test_file = os.path.join(version_dir, "model.onnx")
        else:
            test_file = os.path.join(model_dir, "model.onnx")

        with open(test_file, "w") as f:
            f.write("fake model data")

        config_file = os.path.join(model_dir, "config.pbtxt")
        with open(config_file, "w") as f:
            f.write("fake config")

        return model_dir, test_file, config_file

    def test_generate_config_pbtxt(self):
        """Test generating config.pbtxt for different model types with proper structure."""
        test_cases = [
            (
                "marqo-fashionSigLIP-image-encoder",
                8,
                ["TYPE_FP32"],
                ["[3, 224, 224]", "[768]"],
            ),
            (
                "all-MiniLM-L6-v2-text-encoder",
                16,
                ["TYPE_INT64", "TYPE_FP32"],
                [
                    "[-1]",
                    "[-1, 384]",
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                    "last_hidden_state",
                ],
            ),
        ]

        for model_key, max_batch_size, data_types, expected_strings in test_cases:
            with self.subTest(model=model_key):
                model_props = self._get_model_props(model_key)
                config = ModelManager.generate_config_pbtxt_file(model_props)

                # Verify basic structure
                self.assertIsInstance(config, str)
                self.assertGreater(len(config), 0)
                self.assertIn("name:", config)
                self.assertIn("input [", config)
                self.assertIn("output [", config)

                # Verify model-specific content
                self.assertIn(model_key, config)
                self.assertIn(f"max_batch_size: {max_batch_size}", config)
                for data_type in data_types:
                    self.assertIn(data_type, config)
                for expected_string in expected_strings:
                    self.assertIn(expected_string, config)

    def test_config_pbtxt_properties(self):
        """Test that config.pbtxt generation is deterministic and contains all properties."""
        model_props = self._get_model_props("all-MiniLM-L6-v2-text-encoder")

        # Test deterministic generation
        config1 = ModelManager.generate_config_pbtxt_file(model_props)
        config2 = ModelManager.generate_config_pbtxt_file(model_props)
        self.assertEqual(config1, config2)

        # Test all properties are included
        self.assertIn(model_props.name, config1)
        self.assertIn(f"max_batch_size: {model_props.max_batch_size}", config1)
        for input_def in model_props.input:
            self.assertIn(input_def.name, config1)
            self.assertIn(input_def.data_type.value, config1)
        for output_def in model_props.output:
            self.assertIn(output_def.name, config1)
            self.assertIn(output_def.data_type.value, config1)

    def test_load_model_workflow(self):
        """Test complete load model workflow including downloader and Triton client calls."""
        test_cases = [
            ("marqo-fashionSigLIP-image-encoder", "image encoder"),
            ("all-MiniLM-L6-v2-text-encoder", "text encoder"),
        ]

        for model_key, description in test_cases:
            with self.subTest(model=model_key, description=description):
                self.mock_triton_client.reset_mock()
                model_props = self._get_model_props(model_key)

                with patch(
                    "model_management.services.model_manager.model_manager.TritonModelDownloader"
                ) as mock_downloader_class:
                    mock_downloader = MagicMock()
                    mock_downloader_class.return_value = mock_downloader

                    self.model_manager.load_model(model_props)

                    # Verify downloader was configured correctly
                    call_kwargs = mock_downloader_class.call_args[1]
                    self.assertEqual(model_props.sources, call_kwargs["sources"])
                    self.assertEqual(self.temp_dir, call_kwargs["base_dir"])
                    self.assertEqual(model_props.name, call_kwargs["model_name"])
                    self.assertFalse(call_kwargs["overwrite"])
                    self.assertIn(model_props.name, call_kwargs["config_pbtxt"])

                    # Verify methods were called
                    mock_downloader.prepare_and_download.assert_called_once()
                    self.mock_triton_client.load_model.assert_called_once_with(
                        model_props.name
                    )

    def test_unload_model_with_file_operations(self):
        """Test unloading model with and without file removal."""
        model_name = "test-model"

        # Test unload without removing files
        model_dir, test_file, config_file = self._create_model_files(
            model_name, include_version_dirs=True
        )
        self.model_manager.unload_model(model_name, remove_files=False)
        self.mock_triton_client.unload_model.assert_called_once_with(model_name)
        self.assertTrue(os.path.exists(model_dir))
        self.assertTrue(os.path.exists(test_file))

        # Test unload with removing files
        self.mock_triton_client.reset_mock()
        self.model_manager.unload_model(model_name, remove_files=True)
        self.mock_triton_client.unload_model.assert_called_once_with(model_name)
        self.assertFalse(os.path.exists(model_dir))

        # Test unload nonexistent model with remove_files (should not raise error)
        self.mock_triton_client.reset_mock()
        self.model_manager.unload_model("nonexistent-model", remove_files=True)
        self.mock_triton_client.unload_model.assert_called_once_with(
            "nonexistent-model"
        )

    def test_unload_nested_directory_structure(self):
        """Test that unloading removes nested directory structures completely."""
        model_name = "test-nested-model"
        model_dir = os.path.join(self.temp_dir, model_name)

        # Create nested structure
        for version in ["1", "2"]:
            version_dir = os.path.join(model_dir, version)
            os.makedirs(version_dir, exist_ok=True)
            with open(os.path.join(version_dir, "model.onnx"), "w") as f:
                f.write(f"v{version} model")

        with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
            f.write("config")

        self.assertTrue(os.path.exists(model_dir))
        self.model_manager.unload_model(model_name, remove_files=True)
        self.assertFalse(os.path.exists(model_dir))

    def test_concurrent_operations_locking(self):
        """Test that model operations use locking mechanism correctly."""
        from model_management.services.model_manager import model_manager
        from model_management.services.model_manager.model_manager import (
            _model_op_guard,
        )

        # Verify lock exists
        self.assertIsNotNone(model_manager._MODEL_IO_LOCK)
        self.assertTrue(hasattr(model_manager._MODEL_IO_LOCK, "acquire"))
        self.assertTrue(hasattr(model_manager._MODEL_IO_LOCK, "release"))

        # Test timeout raises error
        test_lock = threading.Lock()
        test_lock.acquire()
        try:
            with self.assertRaises(ModelOperationInProgressError) as context:
                with _model_op_guard(test_lock, timeout=0.1):
                    pass
            self.assertIn(
                "Another model load/unload operation is in progress",
                str(context.exception),
            )
        finally:
            test_lock.release()

    def test_model_operations_sequence(self):
        """Test sequential load and unload operations for multiple models."""
        models = [self._get_model_props(key) for key in test_model_properties.keys()]

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            for model_props in models:
                with self.subTest(model=model_props.name):
                    # Load model
                    self.model_manager.load_model(model_props)
                    self.mock_triton_client.load_model.assert_called_with(
                        model_props.name
                    )

                    # Unload model
                    self.model_manager.unload_model(model_props.name)
                    self.mock_triton_client.unload_model.assert_called_with(
                        model_props.name
                    )

    def test_triton_client_error_propagation(self):
        """Test that Triton client errors are properly propagated."""
        from model_management.services.errors import (
            TritonModelLoadError,
            TritonModelUnloadError,
        )

        model_props = self._get_model_props("marqo-fashionSigLIP-image-encoder")

        # Test load error propagation
        self.mock_triton_client.load_model.side_effect = TritonModelLoadError(
            "Triton error"
        )
        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader_class.return_value = MagicMock()
            with self.assertRaises(TritonModelLoadError):
                self.model_manager.load_model(model_props)

        # Test unload error propagation
        self.mock_triton_client.unload_model.side_effect = TritonModelUnloadError(
            "Triton unload error"
        )
        with self.assertRaises(TritonModelUnloadError):
            self.model_manager.unload_model("test-model")

    def test_logging_behavior(self):
        """Test that model operations are logged correctly."""
        model_props = self._get_model_props("marqo-fashionSigLIP-image-encoder")

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader_class.return_value = MagicMock()

            with patch(
                "model_management.services.model_manager.model_manager.logger"
            ) as mock_logger:
                # Test load logging
                self.model_manager.load_model(model_props)
                mock_logger.info.assert_any_call(
                    f"Loading model: {model_props.model_dump_json()}"
                )
                mock_logger.info.assert_any_call(f"Model loaded: {model_props.name}")

                # Test unload logging
                mock_logger.reset_mock()
                self.model_manager.unload_model("test-model", remove_files=False)
                mock_logger.info.assert_any_call("Unloading model: test-model")
                mock_logger.info.assert_any_call("Model unloaded: test-model")

                # Test file removal logging
                model_dir, _, _ = self._create_model_files("test-model2")
                mock_logger.reset_mock()
                self.model_manager.unload_model("test-model2", remove_files=True)
                mock_logger.info.assert_any_call("Removed model files for: test-model2")

    def test_model_properties_validation(self):
        """Test that real model properties are valid and can create TritonModelProperties."""
        for model_key, model_dict in test_model_properties.items():
            with self.subTest(model=model_key):
                model_props = TritonModelProperties(**model_dict)
                self.assertEqual(model_dict["name"], model_props.name)
                self.assertEqual(model_dict["maxBatchSize"], model_props.max_batch_size)
                self.assertEqual(len(model_dict["sources"]), len(model_props.sources))
                self.assertEqual(len(model_dict["input"]), len(model_props.input))
                self.assertEqual(len(model_dict["output"]), len(model_props.output))


class TestModelManagerEdgeCases(TestCase):
    """Test edge cases and error scenarios for ModelManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_triton_client = MagicMock(spec=TritonClient)
        self.model_manager = ModelManager(
            marqo_model_cache_path=self.temp_dir, triton_client=self.mock_triton_client
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_unload_with_special_model_names(self):
        """Test unloading models with various special character patterns."""
        test_cases = [
            ("", "empty name"),
            ("model-with-dashes", "dashes"),
            ("model_with_underscores", "underscores"),
            ("model.with.dots", "dots"),
            ("model123", "alphanumeric"),
        ]

        for model_name, description in test_cases:
            with self.subTest(description=description):
                self.mock_triton_client.reset_mock()
                self.model_manager.unload_model(model_name, remove_files=False)
                self.mock_triton_client.unload_model.assert_called_once_with(model_name)

    def test_config_generation_with_minimal_properties(self):
        """Test config generation with minimal required properties uses defaults."""
        minimal_props = TritonModelProperties(
            name="minimal-model",
            sources=["s3://bucket/model.onnx"],
            input=[{"name": "input", "dims": [1], "dataType": "TYPE_FP32"}],
            output=[{"name": "output", "dims": [1], "dataType": "TYPE_FP32"}],
        )

        config = ModelManager.generate_config_pbtxt_file(minimal_props)
        self.assertIn("max_batch_size: 8", config)  # Default value
        self.assertIn("minimal-model", config)

    def test_cache_path_handling(self):
        """Test that ModelManager handles cache paths with and without trailing slashes."""
        # With trailing slash
        manager_with_slash = ModelManager(
            marqo_model_cache_path=self.temp_dir + "/",
            triton_client=self.mock_triton_client,
        )
        self.assertEqual(self.temp_dir + "/", manager_with_slash.marqo_model_cache_path)

        # Without trailing slash
        manager_no_slash = ModelManager(
            marqo_model_cache_path=self.temp_dir,
            triton_client=self.mock_triton_client,
        )
        self.assertEqual(self.temp_dir, manager_no_slash.marqo_model_cache_path)

    def test_unload_with_readonly_permissions(self):
        """Test unload with remove_files on read-only directory raises error."""
        if os.name == "nt":
            self.skipTest("Skipping read-only test on Windows")

        model_name = "readonly-model"
        model_dir = os.path.join(self.temp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "model.onnx"), "w") as f:
            f.write("test")

        os.chmod(model_dir, 0o444)
        try:
            with self.assertRaises(PermissionError):
                self.model_manager.unload_model(model_name, remove_files=True)
        finally:
            os.chmod(model_dir, 0o755)


class TestModelManagerRealDownloads(TestCase):
    """Integration tests for ModelManager with REAL model downloads from S3.

    These tests actually download models from S3 and test the full flow.
    They are slower and require network access but provide comprehensive
    integration testing.
    """

    @classmethod
    def setUpClass(cls):
        cls.text_encoder_name = "all-MiniLM-L6-v2-text-encoder"
        cls.image_encoder_name = "marqo-fashionSigLIP-image-encoder"

        cls.text_encoder_props = TritonModelProperties(
            **test_model_properties[cls.text_encoder_name]
        )
        cls.image_encoder_props = TritonModelProperties(
            **test_model_properties[cls.image_encoder_name]
        )

        cls.config = get_config()
        cls.model_manager = cls.config.model_manager

        # Unload models if they are already loaded
        cls.config.model_manager.unload_model(cls.text_encoder_name)
        cls.config.model_manager.unload_model(cls.image_encoder_name)

    def setUp(self):
        super().setUp()
        self.config.model_manager.unload_model(self.text_encoder_name)
        self.config.model_manager.unload_model(self.image_encoder_name)

    def test_load_models_from_s3(self):
        """Test loading both text and image encoder models from S3."""
        test_cases = [
            (self.text_encoder_props, self.text_encoder_name, "text encoder"),
            (self.image_encoder_props, self.image_encoder_name, "image encoder"),
        ]

        for model_props, model_name, description in test_cases:
            with self.subTest(description=description):
                self.model_manager.load_model(model_props)

                returned = requests.get(
                    f"{self.config.model_manager.triton_client.url}/v2/models/{model_name}/config"
                ).json()
                expected_model_config = model_config[model_name]

                for key, value in expected_model_config.items():
                    self.assertEqual(
                        expected_model_config[key],
                        returned[key],
                        f"Mismatch in model config for key: {key}",
                    )
