import threading
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from model_management.schemas.triton_model_properties import TritonModelProperties
from model_management.services.errors import ModelOperationInProgressError
from model_management.services.model_manager.model_manager import (
    ModelManager,
    _model_op_guard,
)
from model_management.services.triton.triton_client import TritonClient


class TestModelOpGuard(TestCase):
    """Test class for _model_op_guard context manager."""

    def test_model_op_guard_acquires_and_releases_lock(self):
        """Test that _model_op_guard acquires and releases lock successfully."""
        lock = threading.Lock()

        with _model_op_guard(lock, timeout=0.1):
            # Lock should be acquired
            self.assertFalse(lock.acquire(blocking=False))

        # Lock should be released after context
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()

    def test_model_op_guard_raises_on_lock_timeout(self):
        """Test that _model_op_guard raises ModelOperationInProgressError when lock cannot be acquired."""
        lock = threading.Lock()
        lock.acquire()  # Pre-acquire the lock

        try:
            with self.assertRaises(ModelOperationInProgressError) as context:
                with _model_op_guard(lock, timeout=0.1):
                    pass

            self.assertIn(
                "Another model load/unload operation is in progress",
                str(context.exception),
            )
        finally:
            lock.release()

    def test_model_op_guard_releases_lock_on_exception(self):
        """Test that _model_op_guard releases lock even when exception occurs."""
        lock = threading.Lock()

        try:
            with _model_op_guard(lock, timeout=0.1):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be released
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()


class TestModelManager(TestCase):
    """Test class for ModelManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_triton_client = Mock(spec=TritonClient)
        self.marqo_model_cache_path = "/tmp/models"
        self.manager = ModelManager(
            marqo_model_cache_path=self.marqo_model_cache_path,
            triton_client=self.mock_triton_client,
        )

        self.valid_model_props = TritonModelProperties(
            name="test-model",
            max_batch_size=8,
            sources=["s3://bucket/model.onnx"],
            input=[{"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}],
            output=[{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        )

    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        manager = ModelManager(
            marqo_model_cache_path="/tmp/test", triton_client=self.mock_triton_client
        )

        self.assertEqual("/tmp/test", manager.marqo_model_cache_path)
        self.assertEqual(self.mock_triton_client, manager.triton_client)

    @patch(
        "model_management.services.model_manager.model_manager.TritonModelDownloader"
    )
    def test_load_model_success(self, mock_downloader_class):
        """Test successful model loading."""
        mock_downloader = Mock()
        mock_downloader.prepare_and_download = Mock()
        mock_downloader_class.return_value = mock_downloader

        self.manager.load_model(self.valid_model_props)

        # Verify downloader was created with correct parameters
        mock_downloader_class.assert_called_once()
        call_kwargs = mock_downloader_class.call_args[1]
        self.assertEqual(self.valid_model_props.sources, call_kwargs["sources"])
        self.assertEqual(self.marqo_model_cache_path, call_kwargs["base_dir"])
        self.assertEqual("test-model", call_kwargs["model_name"])
        self.assertFalse(call_kwargs["overwrite"])
        self.assertIn("test-model", call_kwargs["config_pbtxt"])

        # Verify downloader was called
        mock_downloader.prepare_and_download.assert_called_once()

        # Verify triton client was called
        self.mock_triton_client.load_model.assert_called_once_with("test-model")

    @patch(
        "model_management.services.model_manager.model_manager.TritonModelDownloader"
    )
    def test_load_model_generates_config_pbtxt(self, mock_downloader_class):
        """Test that load_model generates config.pbtxt."""
        mock_downloader = Mock()
        mock_downloader_class.return_value = mock_downloader

        self.manager.load_model(self.valid_model_props)

        call_kwargs = mock_downloader_class.call_args[1]
        config_pbtxt = call_kwargs["config_pbtxt"]

        # Verify config contains expected content
        self.assertIn("test-model", config_pbtxt)
        self.assertIn("8", config_pbtxt)  # max_batch_size
        self.assertIn("input", config_pbtxt)
        self.assertIn("output", config_pbtxt)

    def test_unload_model_without_removing_files(self):
        """Test unload_model without removing files."""
        self.manager.unload_model("test-model", remove_files=False)

        self.mock_triton_client.unload_model.assert_called_once_with("test-model")

    @patch("os.path.exists")
    @patch("os.walk")
    @patch("os.remove")
    @patch("os.rmdir")
    def test_unload_model_with_removing_files(
        self, mock_rmdir, mock_remove, mock_walk, mock_exists
    ):
        """Test unload_model with removing files."""
        mock_exists.return_value = True
        mock_walk.return_value = [
            ("/tmp/models/test-model/1", [], ["model.onnx"]),
            ("/tmp/models/test-model", ["1"], ["config.pbtxt"]),
        ]

        self.manager.unload_model("test-model", remove_files=True)

        # Verify triton client was called
        self.mock_triton_client.unload_model.assert_called_once_with("test-model")

        # Verify files were removed
        self.assertGreater(mock_remove.call_count, 0)
        self.assertGreater(mock_rmdir.call_count, 0)

    @patch("os.path.exists")
    def test_unload_model_when_directory_does_not_exist(self, mock_exists):
        """Test unload_model when model directory doesn't exist."""
        mock_exists.return_value = False

        self.manager.unload_model("test-model", remove_files=True)

        # Should still call triton client
        self.mock_triton_client.unload_model.assert_called_once_with("test-model")

    def test_generate_config_pbtxt_file(self):
        """Test generate_config_pbtxt_file static method."""
        config_pbtxt = ModelManager.generate_config_pbtxt_file(self.valid_model_props)

        # Verify config contains model properties
        self.assertIn("test-model", config_pbtxt)
        self.assertIn("8", config_pbtxt)  # max_batch_size
        self.assertIn("input", config_pbtxt)
        self.assertIn("output", config_pbtxt)
        self.assertIn("TYPE_FP32", config_pbtxt)
        self.assertIsInstance(config_pbtxt, str)
        self.assertGreater(len(config_pbtxt), 0)

    def test_generate_config_pbtxt_file_with_different_models(self):
        """Test generate_config_pbtxt_file with various model configurations."""
        test_cases = [
            {
                "name": "image-model",
                "max_batch_size": 16,
                "sources": ["s3://bucket/model.onnx"],
                "input": [
                    {"name": "image", "dims": [3, 384, 384], "dataType": "TYPE_FP16"}
                ],
                "output": [
                    {"name": "embeddings", "dims": [512], "dataType": "TYPE_FP16"}
                ],
            },
            {
                "name": "text-model",
                "max_batch_size": 32,
                "sources": ["s3://bucket/model.onnx"],
                "input": [{"name": "tokens", "dims": [512], "dataType": "TYPE_INT32"}],
                "output": [
                    {"name": "embeddings", "dims": [768], "dataType": "TYPE_FP32"}
                ],
            },
        ]

        for model_data in test_cases:
            with self.subTest(model_name=model_data["name"]):
                model_props = TritonModelProperties(**model_data)
                config_pbtxt = ModelManager.generate_config_pbtxt_file(model_props)

                self.assertIn(model_data["name"], config_pbtxt)
                self.assertIn(str(model_data["max_batch_size"]), config_pbtxt)

    def test_generate_config_pbtxt_file_with_multiple_inputs(self):
        """Test generate_config_pbtxt_file with multiple inputs."""
        model_props = TritonModelProperties(
            name="multimodal-model",
            sources=["s3://bucket/model.onnx"],
            input=[
                {"name": "image", "dims": [3, 224, 224], "dataType": "TYPE_FP32"},
                {"name": "text", "dims": [77], "dataType": "TYPE_INT64"},
            ],
            output=[{"name": "output", "dims": [512], "dataType": "TYPE_FP32"}],
        )

        config_pbtxt = ModelManager.generate_config_pbtxt_file(model_props)

        self.assertIn("image", config_pbtxt)
        self.assertIn("text", config_pbtxt)
        self.assertIn("TYPE_FP32", config_pbtxt)
        self.assertIn("TYPE_INT64", config_pbtxt)

    @patch(
        "model_management.services.model_manager.model_manager.TritonModelDownloader"
    )
    def test_load_model_with_different_model_properties(self, mock_downloader_class):
        """Test load_model with various model configurations."""
        test_models = [
            TritonModelProperties(
                name="model-1",
                max_batch_size=4,
                sources=["s3://bucket/model-1/model.onnx"],
                input=[{"name": "input", "dims": [1], "dataType": "TYPE_FP32"}],
                output=[{"name": "output", "dims": [1], "dataType": "TYPE_FP32"}],
            ),
            TritonModelProperties(
                name="model-2",
                max_batch_size=64,
                sources=["http://example.com/model-2/model.onnx"],
                input=[{"name": "data", "dims": [512], "dataType": "TYPE_INT32"}],
                output=[{"name": "result", "dims": [256], "dataType": "TYPE_FP16"}],
            ),
        ]

        mock_downloader = Mock()
        mock_downloader_class.return_value = mock_downloader

        for model_props in test_models:
            with self.subTest(model_name=model_props.name):
                self.manager.load_model(model_props)
                self.mock_triton_client.load_model.assert_called_with(model_props.name)

    def test_unload_model_with_various_model_names(self):
        """Test unload_model with various model names."""
        test_cases = [
            "simple-model",
            "model_with_underscores",
            "model-123",
        ]

        for model_name in test_cases:
            with self.subTest(model_name=model_name):
                self.mock_triton_client.reset_mock()
                self.manager.unload_model(model_name, remove_files=False)
                self.mock_triton_client.unload_model.assert_called_once_with(model_name)


class TestGenerateConfigPbtxt(TestCase):
    """Test class for generate_config_pbtxt_file method with fixture comparison."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.fixtures_dir = Path(__file__).parent / "fixtures"

    def _load_fixture(self, filename: str) -> str:
        """Load a fixture file and return its contents."""
        fixture_path = self.fixtures_dir / filename
        return fixture_path.read_text()

    def test_generate_config_pbtxt_simple_model(self):
        """Test generate_config_pbtxt_file matches expected output for simple model."""
        model_props = TritonModelProperties(
            name="simple-model",
            max_batch_size=8,
            sources=["s3://bucket/model.onnx"],
            input=[{"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}],
            output=[{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        )

        generated_config = ModelManager.generate_config_pbtxt_file(model_props)
        expected_config = self._load_fixture("simple_model_config.pbtxt")

        self.assertEqual(expected_config, generated_config)

    def test_generate_config_pbtxt_multimodal_model(self):
        """Test generate_config_pbtxt_file matches expected output for multimodal model."""
        model_props = TritonModelProperties(
            name="multimodal-model",
            max_batch_size=16,
            sources=["s3://bucket/model.onnx"],
            input=[
                {"name": "image", "dims": [3, 224, 224], "dataType": "TYPE_FP32"},
                {"name": "text", "dims": [77], "dataType": "TYPE_INT64"},
            ],
            output=[{"name": "embeddings", "dims": [512], "dataType": "TYPE_FP32"}],
        )

        generated_config = ModelManager.generate_config_pbtxt_file(model_props)
        expected_config = self._load_fixture("multimodal_model_config.pbtxt")

        self.assertEqual(expected_config, generated_config)

    def test_generate_config_pbtxt_text_model(self):
        """Test generate_config_pbtxt_file matches expected output for text model."""
        model_props = TritonModelProperties(
            name="text-model",
            max_batch_size=32,
            sources=["s3://bucket/model.onnx"],
            input=[{"name": "tokens", "dims": [512], "dataType": "TYPE_INT32"}],
            output=[{"name": "embeddings", "dims": [768], "dataType": "TYPE_FP16"}],
        )

        generated_config = ModelManager.generate_config_pbtxt_file(model_props)
        expected_config = self._load_fixture("text_model_config.pbtxt")

        self.assertEqual(expected_config, generated_config)

    def test_generate_config_pbtxt_large_batch_model(self):
        """Test generate_config_pbtxt_file matches expected output for large batch model."""
        model_props = TritonModelProperties(
            name="large-batch-model",
            max_batch_size=128,
            sources=["s3://bucket/model.onnx"],
            input=[
                {"name": "input_data", "dims": [1, 512, 768], "dataType": "TYPE_FP16"}
            ],
            output=[
                {"name": "output_data", "dims": [1, 512, 768], "dataType": "TYPE_FP16"}
            ],
        )

        generated_config = ModelManager.generate_config_pbtxt_file(model_props)
        expected_config = self._load_fixture("large_batch_model_config.pbtxt")

        self.assertEqual(expected_config, generated_config)

    def test_generate_config_pbtxt_contains_required_fields(self):
        """Test that generated config contains all required fields."""
        model_props = TritonModelProperties(
            name="test-model",
            max_batch_size=8,
            sources=["s3://bucket/model.onnx"],
            input=[{"name": "input", "dims": [3, 224, 224], "dataType": "TYPE_FP32"}],
            output=[{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        )

        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Verify required fields are present
        self.assertIn('name: "test-model"', config)
        self.assertIn('backend: "onnxruntime"', config)
        self.assertIn("max_batch_size: 8", config)
        self.assertIn("input [", config)
        self.assertIn("output [", config)
        self.assertIn("dynamic_batching { }", config)

    def test_generate_config_pbtxt_preserves_data_types(self):
        """Test that generated config preserves all data types correctly."""
        test_cases = [
            ("TYPE_FP32", "TYPE_FP32"),
            ("TYPE_FP16", "TYPE_FP16"),
            ("TYPE_INT32", "TYPE_INT32"),
            ("TYPE_INT64", "TYPE_INT64"),
            ("TYPE_BF16", "TYPE_BF16"),
        ]

        for input_type, output_type in test_cases:
            with self.subTest(input_type=input_type, output_type=output_type):
                model_props = TritonModelProperties(
                    name="test-model",
                    sources=["s3://bucket/model.onnx"],
                    input=[{"name": "input", "dims": [1], "dataType": input_type}],
                    output=[{"name": "output", "dims": [1], "dataType": output_type}],
                )

                config = ModelManager.generate_config_pbtxt_file(model_props)

                self.assertIn(f"data_type: {input_type}", config)
                self.assertIn(f"data_type: {output_type}", config)

    def test_generate_config_pbtxt_formats_dimensions_correctly(self):
        """Test that dimensions are formatted correctly in generated config."""
        test_cases = [
            ([1], "[1]"),
            ([3, 224, 224], "[3, 224, 224]"),
            ([1, 512, 768], "[1, 512, 768]"),
            ([768], "[768]"),
        ]

        for dims, expected_format in test_cases:
            with self.subTest(dims=dims):
                model_props = TritonModelProperties(
                    name="test-model",
                    sources=["s3://bucket/model.onnx"],
                    input=[{"name": "input", "dims": dims, "dataType": "TYPE_FP32"}],
                    output=[{"name": "output", "dims": [1], "dataType": "TYPE_FP32"}],
                )

                config = ModelManager.generate_config_pbtxt_file(model_props)
                self.assertIn(f"dims: {expected_format}", config)

    def test_generate_config_pbtxt_handles_multiple_inputs_correctly(self):
        """Test that multiple inputs are formatted correctly with proper comma separation."""
        model_props = TritonModelProperties(
            name="multi-input-model",
            sources=["s3://bucket/model.onnx"],
            input=[
                {"name": "input1", "dims": [3, 224, 224], "dataType": "TYPE_FP32"},
                {"name": "input2", "dims": [512], "dataType": "TYPE_INT32"},
                {"name": "input3", "dims": [77], "dataType": "TYPE_INT64"},
            ],
            output=[{"name": "output", "dims": [768], "dataType": "TYPE_FP32"}],
        )

        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Verify all inputs are present
        self.assertIn('name: "input1"', config)
        self.assertIn('name: "input2"', config)
        self.assertIn('name: "input3"', config)

        # Verify proper formatting with commas between inputs
        lines = config.split("\n")
        input_section = "\n".join(
            lines[lines.index("input [") : lines.index("output [")]
        )

        # Count closing braces followed by commas (indicates proper separation)
        self.assertEqual(2, input_section.count("},"))

    def test_generate_config_pbtxt_single_output_no_comma(self):
        """Test that single output doesn't have trailing comma."""
        model_props = TritonModelProperties(
            name="test-model",
            sources=["s3://bucket/model.onnx"],
            input=[{"name": "input", "dims": [1], "dataType": "TYPE_FP32"}],
            output=[{"name": "output", "dims": [1], "dataType": "TYPE_FP32"}],
        )

        config = ModelManager.generate_config_pbtxt_file(model_props)

        # Find output section
        lines = config.split("\n")
        output_start = lines.index("output [")
        output_section = "\n".join(lines[output_start:])

        # Single output should not have comma after closing brace
        self.assertNotIn("},", output_section)
        self.assertIn("}", output_section)
