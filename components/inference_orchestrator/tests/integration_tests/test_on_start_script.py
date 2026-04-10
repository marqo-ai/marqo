"""Integration tests for the on_start_script module.

These tests verify that the startup script correctly preloads models and performs
initialization tasks. Since these are integration tests, they will actually load
models and perform real inference operations.
"""

from io import StringIO
from unittest.mock import MagicMock, patch

from inference_orchestrator.on_start_script import CacheModels, PrintVersion, on_start
from inference_orchestrator.services.errors import UnsupportedModelError
from inference_orchestrator.services.triton_inference.model_manager import model_manager
from tests.integration_tests.test_case import InferenceTestCase


class TestPrintVersion(InferenceTestCase):
    """Test the PrintVersion startup task."""

    def test_print_version_outputs_version_string(self):
        """Test that PrintVersion prints the version to stdout."""
        print_version = PrintVersion()

        # Capture stdout
        with patch("sys.stdout", new=StringIO()) as fake_out:
            print_version.run()
            output = fake_out.getvalue()

        # Verify version is printed
        self.assertIn("Version:", output)
        # Version should be a non-empty string after "Version: "
        version_part = output.split("Version:")[1].strip()
        self.assertGreater(len(version_part), 0)


class TestCacheModels(InferenceTestCase):
    """Test the CacheModels startup task.

    These tests will actually load models and perform inference, so they may take
    some time to run.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.eject_all_models()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.eject_all_models()

    def setUp(self):
        super().setUp()
        # Eject all models before each test for clean state
        self.eject_all_models()

    def test_cache_models_run_with_empty_preload_list(self):
        """Test that run() completes successfully with no models to preload."""
        # Create a mock settings object with empty preload list
        mock_settings = MagicMock()
        mock_settings.marqo_models_to_preload = []

        with patch("inference_orchestrator.on_start_script.settings", mock_settings):
            cache_models = CacheModels(self.config)

            # Should complete without error
            cache_models.run()

            # No models should be loaded
            loaded_models = model_manager.get_loaded_models()
            self.assertEqual(0, len(loaded_models["models"]))

    def test_cache_models_run_with_multiple_models(self):
        """Test that run() successfully preloads multiple models."""
        models_to_preload = ["hf/all-MiniLM-L6-v2", "hf/e5-base-v2"]

        # Create a mock settings object
        mock_settings = MagicMock()
        mock_settings.marqo_models_to_preload = models_to_preload

        with patch("inference_orchestrator.on_start_script.settings", mock_settings):
            cache_models = CacheModels(self.config)

            # Run the preloading
            cache_models.run()

            # Verify both models were loaded
            loaded_models = model_manager.get_loaded_models()
            # Should have at least as many models as we preloaded
            self.assertGreaterEqual(
                len(loaded_models["models"]), len(models_to_preload)
            )

    def test_cache_models_run_performs_warmup_iterations(self):
        """Test that run() performs multiple warmup iterations for timing."""
        model_name = "hf/all-MiniLM-L6-v2"

        # Create a mock settings object
        mock_settings = MagicMock()
        mock_settings.marqo_models_to_preload = [model_name]

        with patch("inference_orchestrator.on_start_script.settings", mock_settings):
            cache_models = CacheModels(self.config)

            # Mock _preload_model to track calls
            original_preload = cache_models._preload_model
            call_count = {"count": 0}

            def counting_preload(*args, **kwargs):
                call_count["count"] += 1
                return original_preload(*args, **kwargs)

            cache_models._preload_model = counting_preload

            # Run the preloading
            cache_models.run()

            # Verify _preload_model was called 11 times (1 warmup + 10 timed runs)
            self.assertEqual(11, call_count["count"])

    def test_cache_models_load_model_properties_from_registry(self):
        """Test that _load_model_properties_from_model_registry returns properties."""
        cache_models = CacheModels(self.config)
        model_name = "hf/all-MiniLM-L6-v2"

        properties = cache_models._load_model_properties_from_model_registry(model_name)

        # Verify properties is a dict
        self.assertIsInstance(properties, dict)

        # Verify it has expected keys
        self.assertIn("type", properties)
        self.assertIn("dimensions", properties)


class TestOnStart(InferenceTestCase):
    """Test the on_start function that orchestrates startup tasks."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.eject_all_models()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.eject_all_models()

    def test_on_start_runs_all_startup_tasks(self):
        """Test that on_start runs both CacheModels and PrintVersion."""
        # Create a mock settings object
        mock_settings = MagicMock()
        mock_settings.marqo_models_to_preload = []

        with patch("inference_orchestrator.on_start_script.settings", mock_settings):
            # Mock both tasks to track if they were called
            with patch.object(CacheModels, "run") as mock_cache_run:
                with patch.object(PrintVersion, "run") as mock_print_run:
                    # Run on_start
                    on_start(self.config)

                    # Verify both tasks were run
                    mock_cache_run.assert_called_once()
                    mock_print_run.assert_called_once()

    def test_on_start_creates_correct_task_instances(self):
        """Test that on_start creates CacheModels with correct config."""
        # Create a mock settings object
        mock_settings = MagicMock()
        mock_settings.marqo_models_to_preload = []

        with patch("inference_orchestrator.on_start_script.settings", mock_settings):
            with patch.object(CacheModels, "__init__", return_value=None) as mock_init:
                with patch.object(CacheModels, "run"):
                    with patch.object(PrintVersion, "run"):
                        # Run on_start
                        on_start(self.config)

                        # Verify CacheModels was initialized with the config
                        mock_init.assert_called_once_with(self.config)

    def test_on_start_executes_tasks_in_order(self):
        """Test that on_start executes tasks in the correct order."""
        # Create a mock settings object
        mock_settings = MagicMock()
        mock_settings.marqo_models_to_preload = []

        with patch("inference_orchestrator.on_start_script.settings", mock_settings):
            execution_order = []

            def cache_models_run(self):
                execution_order.append("CacheModels")

            def print_version_run(self):
                execution_order.append("PrintVersion")

            with patch.object(CacheModels, "run", cache_models_run):
                with patch.object(PrintVersion, "run", print_version_run):
                    # Run on_start
                    on_start(self.config)

                    # Verify execution order
                    self.assertEqual(["CacheModels", "PrintVersion"], execution_order)

    def test_on_start_with_actual_model_preloading(self):
        """Test on_start with actual model preloading (full integration test).

        This test actually loads a model and verifies the full startup flow.
        """
        model_name = "hf/all-MiniLM-L6-v2"

        # Clear any loaded models
        self.eject_all_models()

        # Create a mock settings object
        mock_settings = MagicMock()
        mock_settings.marqo_models_to_preload = [model_name]

        with patch("inference_orchestrator.on_start_script.settings", mock_settings):
            # Capture stdout to verify PrintVersion
            with patch("sys.stdout", new=StringIO()) as fake_out:
                # Run on_start
                on_start(self.config)

                # Verify version was printed
                output = fake_out.getvalue()
                self.assertIn("Version:", output)

            # Verify model was loaded
            loaded_models = model_manager.get_loaded_models()
            self.assertGreater(len(loaded_models["models"]), 0)

            model_names = [model["modelName"] for model in loaded_models["models"]]
            self.assertTrue(any(model_name in name for name in model_names))

    def test_a_proper_error_is_raised_when_model_is_not_supported(self):
        """Test that an appropriate error is raised when an unsupported model is specified."""
        unsupported_model_name = "hf/unsupported-model-xyz"

        # Create a mock settings object
        mock_settings = MagicMock()
        mock_settings.marqo_models_to_preload = [unsupported_model_name]

        with patch("inference_orchestrator.on_start_script.settings", mock_settings):
            # Run on_start and expect an error
            with self.assertRaises(UnsupportedModelError) as context:
                on_start(self.config)

            # Verify the error message indicates unsupported model
            self.assertIn("not supported", str(context.exception))


class TestCacheModelsEdgeCases(InferenceTestCase):
    """Test edge cases and error scenarios for CacheModels."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.eject_all_models()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.eject_all_models()

    def test_cache_models_with_dict_model_missing_keys_raises_error(self):
        """Test that preload with incomplete dict config raises error."""
        cache_models = CacheModels(self.config)

        # Dict missing 'modelProperties'
        invalid_model_dict = {"model": "hf/all-MiniLM-L6-v2"}

        # Should raise KeyError or similar
        with self.assertRaises(Exception):
            cache_models._preload_model(model=invalid_model_dict, content="test")
