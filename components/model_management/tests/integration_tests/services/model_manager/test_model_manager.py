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

    def test_model_manager_initialization(self):
        """Test that ModelManager initializes with correct configuration."""
        self.assertEqual(self.temp_dir, self.model_manager.marqo_model_cache_path)
        self.assertIs(self.mock_triton_client, self.model_manager.triton_client)

    def test_generate_config_pbtxt_for_image_encoder(self):
        """Test generating config.pbtxt for image encoder model."""
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Verify config contains essential elements
        self.assertIn("marqo-fashionSigLIP-image-encoder", config)
        self.assertIn("max_batch_size: 8", config)
        self.assertIn("input", config)
        self.assertIn("output", config)
        self.assertIn("TYPE_FP32", config)
        # Dimensions might be formatted as [3, 224, 224] without spaces
        self.assertIn("[3, 224, 224]", config)
        self.assertIn("[768]", config)

    def test_generate_config_pbtxt_for_text_encoder(self):
        """Test generating config.pbtxt for text encoder model."""
        model_props = TritonModelProperties(
            **test_model_properties["all-MiniLM-L6-v2-text-encoder"]
        )

        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Verify config contains essential elements
        self.assertIn("all-MiniLM-L6-v2-text-encoder", config)
        self.assertIn("max_batch_size: 16", config)
        self.assertIn("input_ids", config)
        self.assertIn("attention_mask", config)
        self.assertIn("token_type_ids", config)
        self.assertIn("last_hidden_state", config)
        self.assertIn("TYPE_INT64", config)
        self.assertIn("TYPE_FP32", config)

    def test_generate_config_pbtxt_with_dynamic_dimensions(self):
        """Test that config.pbtxt correctly handles dynamic dimensions (-1)."""
        model_props = TritonModelProperties(
            **test_model_properties["all-MiniLM-L6-v2-text-encoder"]
        )

        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Dynamic dimensions should be represented as -1
        # Format might be [-1] without spaces
        self.assertIn("[-1]", config)
        self.assertIn("[-1, 384]", config)

    def test_generate_config_pbtxt_structure(self):
        """Test that generated config.pbtxt has proper structure."""
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Verify basic structure
        self.assertIsInstance(config, str)
        self.assertGreater(len(config), 0)

        # Verify it looks like a protobuf text format
        self.assertIn("name:", config)
        self.assertIn("input [", config)
        self.assertIn("output [", config)

    def test_load_model_calls_triton_client(self):
        """Test that load_model calls TritonClient.load_model."""
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        # Mock the downloader to avoid actual S3 access
        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            self.model_manager.load_model(model_props)

            # Verify downloader was called with correct parameters
            mock_downloader_class.assert_called_once()
            call_kwargs = mock_downloader_class.call_args[1]
            self.assertEqual(model_props.sources, call_kwargs["sources"])
            self.assertEqual(self.temp_dir, call_kwargs["base_dir"])
            self.assertEqual(model_props.name, call_kwargs["model_name"])
            self.assertFalse(call_kwargs["overwrite"])

            # Verify prepare_and_download was called
            mock_downloader.prepare_and_download.assert_called_once()

            # Verify Triton client was called to load the model
            self.mock_triton_client.load_model.assert_called_once_with(model_props.name)

    def test_load_model_with_different_models(self):
        """Test loading different model types."""
        test_cases = [
            ("marqo-fashionSigLIP-image-encoder", "image encoder model"),
            ("all-MiniLM-L6-v2-text-encoder", "text encoder model"),
        ]

        for model_key, description in test_cases:
            with self.subTest(model=model_key, description=description):
                self.mock_triton_client.reset_mock()
                model_props = TritonModelProperties(**test_model_properties[model_key])

                with patch(
                    "model_management.services.model_manager.model_manager.TritonModelDownloader"
                ) as mock_downloader_class:
                    mock_downloader = MagicMock()
                    mock_downloader_class.return_value = mock_downloader

                    self.model_manager.load_model(model_props)

                    # Verify triton client was called
                    self.mock_triton_client.load_model.assert_called_once_with(
                        model_props.name
                    )

    def test_unload_model_calls_triton_client(self):
        """Test that unload_model calls TritonClient.unload_model."""
        model_name = "test-model"

        self.model_manager.unload_model(model_name, remove_files=False)

        self.mock_triton_client.unload_model.assert_called_once_with(model_name)

    def test_unload_model_without_removing_files(self):
        """Test unloading a model without removing files."""
        model_name = "marqo-fashionSigLIP-image-encoder"

        # Create a fake model directory
        model_dir = os.path.join(self.temp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        test_file = os.path.join(model_dir, "model.onnx")
        with open(test_file, "w") as f:
            f.write("fake model data")

        self.model_manager.unload_model(model_name, remove_files=False)

        # Verify Triton client was called
        self.mock_triton_client.unload_model.assert_called_once_with(model_name)

        # Verify files still exist
        self.assertTrue(os.path.exists(model_dir))
        self.assertTrue(os.path.exists(test_file))

    def test_unload_model_with_removing_files(self):
        """Test unloading a model and removing its files."""
        model_name = "marqo-fashionSigLIP-image-encoder"

        # Create a fake model directory with files
        model_dir = os.path.join(self.temp_dir, model_name)
        version_dir = os.path.join(model_dir, "1")
        os.makedirs(version_dir, exist_ok=True)

        test_file = os.path.join(version_dir, "model.onnx")
        config_file = os.path.join(model_dir, "config.pbtxt")

        with open(test_file, "w") as f:
            f.write("fake model data")
        with open(config_file, "w") as f:
            f.write("fake config")

        self.model_manager.unload_model(model_name, remove_files=True)

        # Verify Triton client was called
        self.mock_triton_client.unload_model.assert_called_once_with(model_name)

        # Verify files were removed
        self.assertFalse(os.path.exists(model_dir))
        self.assertFalse(os.path.exists(test_file))
        self.assertFalse(os.path.exists(config_file))

    def test_unload_nonexistent_model_with_remove_files(self):
        """Test unloading a model that doesn't have files on disk."""
        model_name = "nonexistent-model"

        # Model directory doesn't exist
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, model_name)))

        # Should not raise error
        self.model_manager.unload_model(model_name, remove_files=True)

        self.mock_triton_client.unload_model.assert_called_once_with(model_name)

    def test_load_model_with_overwrite_disabled(self):
        """Test that load_model sets overwrite=False for downloader."""
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            self.model_manager.load_model(model_props)

            # Verify overwrite is False
            call_kwargs = mock_downloader_class.call_args[1]
            self.assertFalse(call_kwargs["overwrite"])

    def test_model_manager_with_real_directory_structure(self):
        """Test ModelManager creates proper directory structure."""
        model_name = "test-model"
        model_dir = os.path.join(self.temp_dir, model_name)

        # Create model directory as the downloader would
        os.makedirs(model_dir, exist_ok=True)

        # Verify directory was created
        self.assertTrue(os.path.exists(model_dir))

        # Test unload with remove
        self.model_manager.unload_model(model_name, remove_files=True)

        # Verify directory was removed
        self.assertFalse(os.path.exists(model_dir))

    def test_concurrent_operations_use_lock(self):
        """Test that model operations use a lock mechanism."""
        # Test that the lock exists and has acquire/release methods
        from model_management.services.model_manager import model_manager

        # Verify the lock exists and is a threading lock
        self.assertIsNotNone(model_manager._MODEL_IO_LOCK)
        self.assertTrue(hasattr(model_manager._MODEL_IO_LOCK, "acquire"))
        self.assertTrue(hasattr(model_manager._MODEL_IO_LOCK, "release"))

        # Test that operations work with the lock
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            # This should complete successfully with the lock
            self.model_manager.load_model(model_props)

            # Verify operation completed
            self.mock_triton_client.load_model.assert_called_once_with(model_props.name)

    def test_model_op_guard_timeout_raises_error(self):
        """Test that model_op_guard raises error when lock cannot be acquired."""
        from model_management.services.model_manager.model_manager import (
            _model_op_guard,
        )

        # Create a lock and hold it
        test_lock = threading.Lock()
        test_lock.acquire()

        try:
            # Try to acquire with timeout - should raise error
            with self.assertRaises(ModelOperationInProgressError) as context:
                with _model_op_guard(test_lock, timeout=0.1):
                    pass

            self.assertIn(
                "Another model load/unload operation is in progress",
                str(context.exception),
            )
        finally:
            test_lock.release()

    def test_load_then_unload_sequence(self):
        """Test loading and then unloading a model in sequence."""
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            # Load model
            self.model_manager.load_model(model_props)
            self.mock_triton_client.load_model.assert_called_once_with(model_props.name)

            # Unload model
            self.model_manager.unload_model(model_props.name, remove_files=False)
            self.mock_triton_client.unload_model.assert_called_once_with(
                model_props.name
            )

    def test_multiple_sequential_operations(self):
        """Test multiple sequential load/unload operations."""
        models = [
            TritonModelProperties(
                **test_model_properties["marqo-fashionSigLIP-image-encoder"]
            ),
            TritonModelProperties(
                **test_model_properties["all-MiniLM-L6-v2-text-encoder"]
            ),
        ]

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            for model_props in models:
                with self.subTest(model=model_props.name):
                    # Load model
                    self.model_manager.load_model(model_props)

                    # Verify triton client was called
                    self.mock_triton_client.load_model.assert_called_with(
                        model_props.name
                    )

                    # Unload model
                    self.model_manager.unload_model(model_props.name)

                    # Verify triton client was called
                    self.mock_triton_client.unload_model.assert_called_with(
                        model_props.name
                    )

    def test_unload_multiple_models(self):
        """Test unloading multiple different models."""
        model_names = [
            "marqo-fashionSigLIP-image-encoder",
            "all-MiniLM-L6-v2-text-encoder",
            "custom-model-1",
        ]

        for model_name in model_names:
            with self.subTest(model=model_name):
                self.mock_triton_client.reset_mock()

                self.model_manager.unload_model(model_name, remove_files=False)

                self.mock_triton_client.unload_model.assert_called_once_with(model_name)

    def test_model_manager_cache_path_handling(self):
        """Test that ModelManager correctly handles cache path."""
        # Test with trailing slash
        manager_with_slash = ModelManager(
            marqo_model_cache_path=self.temp_dir + "/",
            triton_client=self.mock_triton_client,
        )
        self.assertEqual(self.temp_dir + "/", manager_with_slash.marqo_model_cache_path)

        # Test with no trailing slash
        manager_no_slash = ModelManager(
            marqo_model_cache_path=self.temp_dir, triton_client=self.mock_triton_client
        )
        self.assertEqual(self.temp_dir, manager_no_slash.marqo_model_cache_path)

    def test_generate_config_pbtxt_is_deterministic(self):
        """Test that config generation is deterministic for the same input."""
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        config1 = ModelManager.generate_config_pbtxt_file(model_props)
        config2 = ModelManager.generate_config_pbtxt_file(model_props)

        self.assertEqual(config1, config2)

    def test_config_pbtxt_contains_all_model_properties(self):
        """Test that config.pbtxt contains all required model properties."""
        model_props = TritonModelProperties(
            **test_model_properties["all-MiniLM-L6-v2-text-encoder"]
        )

        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Check model name
        self.assertIn(model_props.name, config)

        # Check max batch size
        self.assertIn(f"max_batch_size: {model_props.max_batch_size}", config)

        # Check all inputs
        for input_def in model_props.input:
            self.assertIn(input_def.name, config)
            self.assertIn(input_def.data_type.value, config)

        # Check all outputs
        for output_def in model_props.output:
            self.assertIn(output_def.name, config)
            self.assertIn(output_def.data_type.value, config)

    def test_unload_with_nested_directory_structure(self):
        """Test unloading removes nested directory structure."""
        model_name = "test-nested-model"

        # Create nested directory structure
        model_dir = os.path.join(self.temp_dir, model_name)
        version1_dir = os.path.join(model_dir, "1")
        version2_dir = os.path.join(model_dir, "2")
        os.makedirs(version1_dir, exist_ok=True)
        os.makedirs(version2_dir, exist_ok=True)

        # Create files in different directories
        with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
            f.write("config")
        with open(os.path.join(version1_dir, "model.onnx"), "w") as f:
            f.write("v1 model")
        with open(os.path.join(version2_dir, "model.onnx"), "w") as f:
            f.write("v2 model")

        # Verify structure exists
        self.assertTrue(os.path.exists(version1_dir))
        self.assertTrue(os.path.exists(version2_dir))

        # Unload with remove files
        self.model_manager.unload_model(model_name, remove_files=True)

        # Verify entire structure is removed
        self.assertFalse(os.path.exists(model_dir))

    def test_load_model_generates_config_pbtxt(self):
        """Test that load_model generates config.pbtxt for the downloader."""
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            self.model_manager.load_model(model_props)

            # Verify config_pbtxt was passed to downloader
            call_kwargs = mock_downloader_class.call_args[1]
            config_pbtxt = call_kwargs["config_pbtxt"]

            # Verify it's a valid config
            self.assertIsInstance(config_pbtxt, str)
            self.assertIn(model_props.name, config_pbtxt)
            self.assertGreater(len(config_pbtxt), 0)

    def test_triton_client_is_required(self):
        """Test that ModelManager requires a TritonClient instance."""
        # This should work fine
        manager = ModelManager(
            marqo_model_cache_path=self.temp_dir, triton_client=self.mock_triton_client
        )

        self.assertIsNotNone(manager.triton_client)

    def test_model_manager_with_real_model_properties_validation(self):
        """Test that real model properties are valid TritonModelProperties."""
        for model_key, model_dict in test_model_properties.items():
            with self.subTest(model=model_key):
                # This should not raise validation errors
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

    def test_unload_model_with_empty_name(self):
        """Test unloading model with empty name."""
        # Empty name should still call triton client (let it handle validation)
        self.model_manager.unload_model("", remove_files=False)

        self.mock_triton_client.unload_model.assert_called_once_with("")

    def test_unload_model_with_special_characters_in_name(self):
        """Test unloading models with special characters in names."""
        test_cases = [
            "model-with-dashes",
            "model_with_underscores",
            "model.with.dots",
            "model123",
        ]

        for model_name in test_cases:
            with self.subTest(model_name=model_name):
                self.mock_triton_client.reset_mock()

                self.model_manager.unload_model(model_name, remove_files=False)

                self.mock_triton_client.unload_model.assert_called_once_with(model_name)

    def test_unload_with_remove_files_on_readonly_dir(self):
        """Test unload with remove_files when directory permissions are restrictive."""
        model_name = "readonly-model"
        model_dir = os.path.join(self.temp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        test_file = os.path.join(model_dir, "model.onnx")
        with open(test_file, "w") as f:
            f.write("test")

        # Make directory read-only (skip on Windows)
        if os.name != "nt":
            os.chmod(model_dir, 0o444)

            try:
                # This should raise a permission error
                with self.assertRaises(PermissionError):
                    self.model_manager.unload_model(model_name, remove_files=True)
            finally:
                # Restore permissions for cleanup
                os.chmod(model_dir, 0o755)

    def test_config_generation_with_minimal_model_properties(self):
        """Test config generation with minimal required properties."""
        minimal_props = {
            "name": "minimal-model",
            "sources": ["s3://bucket/model.onnx"],
            "input": [{"name": "input", "dims": [1], "dataType": "TYPE_FP32"}],
            "output": [{"name": "output", "dims": [1], "dataType": "TYPE_FP32"}],
        }

        model_props = TritonModelProperties(**minimal_props)
        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Should use default max_batch_size of 8
        self.assertIn("max_batch_size: 8", config)
        self.assertIn("minimal-model", config)

    def test_load_model_with_triton_client_error(self):
        """Test that errors from TritonClient are propagated."""
        from model_management.services.errors import TritonModelLoadError

        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        self.mock_triton_client.load_model.side_effect = TritonModelLoadError(
            "Triton error"
        )

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            # Error from Triton should be raised
            with self.assertRaises(TritonModelLoadError):
                self.model_manager.load_model(model_props)

    def test_unload_model_with_triton_client_error(self):
        """Test that errors from TritonClient during unload are propagated."""
        from model_management.services.errors import TritonModelUnloadError

        self.mock_triton_client.unload_model.side_effect = TritonModelUnloadError(
            "Triton unload error"
        )

        # Error should be raised
        with self.assertRaises(TritonModelUnloadError):
            self.model_manager.unload_model("test-model")


class TestModelManagerLogging(TestCase):
    """Test logging behavior of ModelManager."""

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

    def test_load_model_logs_model_properties(self):
        """Test that load_model logs the model properties."""
        model_props = TritonModelProperties(
            **test_model_properties["marqo-fashionSigLIP-image-encoder"]
        )

        with patch(
            "model_management.services.model_manager.model_manager.TritonModelDownloader"
        ) as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            with patch(
                "model_management.services.model_manager.model_manager.logger"
            ) as mock_logger:
                self.model_manager.load_model(model_props)

                # Verify logging was called
                mock_logger.info.assert_any_call(
                    f"Loading model: {model_props.model_dump_json()}"
                )
                mock_logger.info.assert_any_call(f"Model loaded: {model_props.name}")

    def test_unload_model_logs_operations(self):
        """Test that unload_model logs its operations."""
        model_name = "test-model"

        with patch(
            "model_management.services.model_manager.model_manager.logger"
        ) as mock_logger:
            self.model_manager.unload_model(model_name, remove_files=False)

            # Verify logging
            mock_logger.info.assert_any_call(f"Unloading model: {model_name}")
            mock_logger.info.assert_any_call(f"Model unloaded: {model_name}")

    def test_unload_with_file_removal_logs_correctly(self):
        """Test that file removal is logged."""
        model_name = "test-model"
        model_dir = os.path.join(self.temp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "test.txt"), "w") as f:
            f.write("test")

        with patch(
            "model_management.services.model_manager.model_manager.logger"
        ) as mock_logger:
            self.model_manager.unload_model(model_name, remove_files=True)

            # Verify file removal was logged
            mock_logger.info.assert_any_call(f"Removed model files for: {model_name}")


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

    def test_load_text_encoder_model(self):
        """Test loading the text encoder model from S3."""
        self.model_manager.load_model(self.text_encoder_props)

        returned = requests.get(
            f"{self.config.model_manager.triton_client.url}/v2/models/{self.text_encoder_name}/config"
        ).json()
        expected_model_config = model_config[self.text_encoder_name]

        for key, value in expected_model_config.items():
            self.assertEqual(
                returned[key], value, f"Mismatch in model config for key: {key}"
            )

    def test_load_image_encoder_model(self):
        """Test loading the image encoder model from S3."""
        self.model_manager.load_model(self.image_encoder_props)

        returned = requests.get(
            f"{self.config.model_manager.triton_client.url}/v2/models/{self.image_encoder_name}/config"
        ).json()
        expected_model_config = model_config[self.image_encoder_name]

        for key, value in expected_model_config.items():
            self.assertEqual(
                returned[key], value, f"Mismatch in model config for key: {key}"
            )
