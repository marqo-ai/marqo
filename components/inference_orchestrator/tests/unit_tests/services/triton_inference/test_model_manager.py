import threading
import time
from unittest import TestCase
from unittest.mock import Mock, patch

from inference_orchestrator.services.errors import (
    InvalidModelPropertiesError,
    ModelOperationInProgressError,
)
from inference_orchestrator.services.triton_inference.model_manager import model_manager


class TestModelManager(TestCase):
    def setUp(self):
        """Clear the model cache before each test"""
        model_manager.clear_loaded_models()

    def tearDown(self):
        """Clear the model cache after each test"""
        model_manager.clear_loaded_models()

    def test_create_model_cache_key(self):
        """Test that model cache keys are created correctly.
        The hash suffix is hardcoded to detect changes in the key generation algorithm"""
        test_cases = [
            (
                "model1",
                {"name": "test", "dimensions": 512, "type": "clip", "tokens": 77},
                "model1||c8e6",
            ),
            (
                "model2",
                {"name": "bert", "dimensions": 768, "type": "hf"},
                "model2||c663",
            ),
            ("model3", {}, "model3||6e46"),
        ]

        for model_name, model_properties, expected_key in test_cases:
            with self.subTest(model_name=model_name):
                result = model_manager._create_model_cache_key(
                    model_name, model_properties
                )
                self.assertEqual(expected_key, result)

    def test_validate_model_properties_dimension_valid(self):
        """Test that valid dimensions pass validation"""
        valid_cases = [1, 128, 512, 768, 1024]

        for dimensions in valid_cases:
            with self.subTest(dimensions=dimensions):
                # Should not raise an exception
                model_manager._validate_model_properties_dimension(dimensions)

    def test_validate_model_properties_dimension_invalid(self):
        """Test that invalid dimensions raise InvalidModelPropertiesError"""
        invalid_cases = [
            (None, "dimensions must be a positive integer"),
            (0, "dimensions must be a positive integer"),
            (-1, "dimensions must be a positive integer"),
            ("512", "dimensions must be a positive integer"),
            (3.14, "dimensions must be a positive integer"),
        ]

        for dimensions, expected_message in invalid_cases:
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    model_manager._validate_model_properties_dimension(dimensions)
                self.assertIn("dimensions", str(context.exception).lower())
                self.assertIn("positive integer", str(context.exception).lower())

    def test_get_available_models_empty(self):
        """Test that get_available_models returns empty dict when no models loaded"""
        result = model_manager.get_available_models()
        self.assertEqual({}, result)
        self.assertEqual(0, len(result))

    def test_get_available_models_with_models(self):
        """Test that get_available_models returns loaded models"""
        # Manually add some mock models to the cache
        mock_model1 = Mock()
        mock_model2 = Mock()

        model_manager._available_models["model1||test||512||clip||77||"] = mock_model1
        model_manager._available_models["model2||bert||768||hf||||"] = mock_model2

        result = model_manager.get_available_models()

        self.assertEqual(2, len(result))
        self.assertIn("model1||test||512||clip||77||", result)
        self.assertIn("model2||bert||768||hf||||", result)
        self.assertIs(mock_model1, result["model1||test||512||clip||77||"])
        self.assertIs(mock_model2, result["model2||bert||768||hf||||"])

    def test_clear_loaded_models(self):
        """Test that clear_loaded_models empties the cache"""
        # Add some mock models
        model_manager._available_models["key1"] = Mock()
        model_manager._available_models["key2"] = Mock()
        self.assertEqual(2, len(model_manager._available_models))

        # Clear the cache
        model_manager.clear_loaded_models()

        # Verify it's empty
        self.assertEqual(0, len(model_manager._available_models))
        self.assertEqual({}, model_manager._available_models)

    def test_get_loaded_models_empty(self):
        """Test get_loaded_models with no models"""
        result = model_manager.get_loaded_models()

        self.assertIn("models", result)
        self.assertEqual([], result["models"])

    def test_get_loaded_models_non_detailed(self):
        """Test get_loaded_models returns model names without details"""
        mock_model1 = Mock()
        mock_model2 = Mock()

        model_manager._available_models["model1||test"] = mock_model1
        model_manager._available_models["model2||bert"] = mock_model2

        result = model_manager.get_loaded_models(detailed=False)

        self.assertIn("models", result)
        self.assertEqual(2, len(result["models"]))
        self.assertIn(
            {"modelName": "model1||test"},
            result["models"],
        )
        self.assertIn(
            {"modelName": "model2||bert"},
            result["models"],
        )

    def test_get_loaded_models_detailed(self):
        """Test get_loaded_models returns model names with properties"""
        mock_model1 = Mock()
        mock_model1.model_properties.model_dump_json.return_value = (
            '{"name": "test", "dimensions": 512}'
        )

        mock_model2 = Mock()
        mock_model2.model_properties.model_dump_json.return_value = (
            '{"name": "bert", "dimensions": 768}'
        )

        model_manager._available_models["model1||test||512||clip||77||"] = mock_model1
        model_manager._available_models["model2||bert||768||hf||||"] = mock_model2

        result = model_manager.get_loaded_models(detailed=True)

        self.assertIn("models", result)
        self.assertEqual(2, len(result["models"]))

        # Check that model_properties were called with by_alias=True
        mock_model1.model_properties.model_dump_json.assert_called_once_with(
            by_alias=True
        )
        mock_model2.model_properties.model_dump_json.assert_called_once_with(
            by_alias=True
        )

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_manager.get_model_loader"
    )
    def test_update_available_models_new_model(self, mock_get_model_loader):
        """Test that _update_available_models loads new models"""
        mock_triton_client = Mock()
        mock_management_client = Mock()
        mock_model = Mock()
        mock_loader = Mock(return_value=mock_model)
        mock_get_model_loader.return_value = mock_loader

        model_properties = {"name": "test", "dimensions": 512, "type": "open_clip"}
        model_cache_key = "test-model||test||512||clip||||"

        # Ensure the model is not in the cache
        self.assertNotIn(model_cache_key, model_manager._available_models)

        model_manager._update_available_models(
            model_cache_key,
            model_properties,
            triton_client=mock_triton_client,
            model_management_client=mock_management_client,
        )

        # Verify the model was loaded
        self.assertIn(model_cache_key, model_manager._available_models)
        mock_get_model_loader.assert_called_once_with(model_properties)
        mock_loader.assert_called_once_with(
            model_properties=model_properties,
            model_management_client=mock_management_client,
            triton_client=mock_triton_client,
        )
        mock_model.load.assert_called_once()

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_manager.get_model_loader"
    )
    def test_update_available_models_existing_model(self, mock_get_model_loader):
        """Test that _update_available_models does not reload existing models"""
        mock_existing_model = Mock()
        model_cache_key = "test-model||test"

        # Add the model to the cache
        model_manager._available_models[model_cache_key] = mock_existing_model

        model_manager._update_available_models(
            model_cache_key,
            dict(),
            triton_client=Mock(),
            model_management_client=Mock(),
        )

        # Verify get_model_loader was not called (no new model loaded)
        mock_get_model_loader.assert_not_called()

        # Verify the existing model is still in the cache
        self.assertIs(
            mock_existing_model, model_manager._available_models[model_cache_key]
        )

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_manager.get_model_loader"
    )
    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_manager._update_available_models"
    )
    def test_load_model_success(self, mock_update, mock_get_loader):
        """Test load_model successfully loads and returns a model"""
        mock_triton_client = Mock()
        mock_management_client = Mock()
        mock_model = Mock()

        model_name = "test-model"
        model_properties = {"name": "test", "dimensions": 512, "type": "open_clip"}
        model_cache_key = "test-model||f2c6"

        # Pre-populate the cache (simulating what _update_available_models does)
        model_manager._available_models[model_cache_key] = mock_model

        result = model_manager.load_model(
            model_name,
            model_properties,
            triton_client=mock_triton_client,
            model_management_client=mock_management_client,
        )

        # Verify _update_available_models was called with correct parameters
        mock_update.assert_called_once_with(
            model_cache_key,
            model_properties,
            triton_client=mock_triton_client,
            model_management_client=mock_management_client,
        )

        # Verify the correct model was returned
        self.assertIs(mock_model, result)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_manager.get_model_loader"
    )
    def test_eject_model_success(self, mock_get_loader):
        """Test eject_model removes a model from the cache"""
        mock_model = Mock()
        model_cache_key = "test-model||dfsc"

        # Add the model to the cache
        model_manager._available_models[model_cache_key] = mock_model

        result = model_manager.eject_model("test-model||dfsc")

        # Verify the model was unloaded
        mock_model.unload.assert_called_once()

        # Verify the model was removed from the cache
        self.assertNotIn(model_cache_key, model_manager._available_models)

        # Verify the success response
        self.assertEqual("success", result["result"])
        self.assertIn("test-model||dfsc", result["message"])

    def test_eject_model_not_found(self):
        """Test eject_model when model is not in cache"""
        # Try to eject a model that doesn't exist
        result = model_manager.eject_model("nonexistent-model")

        # Should still return success (no error raised)
        self.assertEqual("success", result["result"])

    def test_eject_model_partial_match(self):
        """Test eject_model only removes the first matching model"""
        mock_model1 = Mock()
        mock_model2 = Mock()

        # Add two models with the same prefix
        model_manager._available_models["test1||dfdc"] = mock_model1
        model_manager._available_models["test2||sads"] = mock_model2

        model_manager.eject_model("test1||dfdc")

        # Only the first model should be ejected
        mock_model1.unload.assert_called_once()
        mock_model2.unload.assert_not_called()

        # Only the first model should be removed
        self.assertNotIn("test1||dfdc", model_manager._available_models)
        self.assertIn("test2||sads", model_manager._available_models)

    def test_model_op_guard_success(self):
        """Test _model_op_guard successfully acquires and releases lock"""
        test_lock = threading.Lock()

        with model_manager._model_op_guard(test_lock):
            # Lock should be acquired
            # Try to acquire again (should fail immediately)
            acquired = test_lock.acquire(blocking=False)
            self.assertFalse(acquired)

        # After context exits, lock should be released
        acquired = test_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        test_lock.release()

    def test_model_op_guard_timeout(self):
        """Test _model_op_guard raises error when lock cannot be acquired"""
        test_lock = threading.Lock()

        # Acquire the lock first
        test_lock.acquire()

        try:
            # Try to acquire the lock with a short timeout
            with self.assertRaises(ModelOperationInProgressError) as context:
                with model_manager._model_op_guard(test_lock, timeout=0.1):
                    pass

            self.assertIn("in progress", str(context.exception).lower())
        finally:
            test_lock.release()

    def test_model_op_guard_releases_lock_on_exception(self):
        """Test _model_op_guard releases lock even when exception occurs"""
        test_lock = threading.Lock()

        with self.assertRaises(ValueError):
            with model_manager._model_op_guard(test_lock):
                raise ValueError("Test exception")

        # Lock should still be released
        acquired = test_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        test_lock.release()

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_manager._update_available_models"
    )
    def test_load_model_concurrent_operations_blocked(self, mock_update):
        """Test that concurrent load_model calls are blocked by the lock"""
        mock_triton_client = Mock()
        mock_management_client = Mock()

        # Make _update_available_models slow (longer than lock timeout) to simulate a long operation
        def slow_update(*args, **kwargs):
            time.sleep(0.05)
            model_cache_key = args[0]
            model_manager._available_models[model_cache_key] = Mock()

        mock_update.side_effect = slow_update

        results = []
        errors = []

        def load_in_thread():
            try:
                result = model_manager.load_model(
                    "test-model",
                    {"name": "test", "dimensions": 512},
                    triton_client=mock_triton_client,
                    model_management_client=mock_management_client,
                    timeout=0.01,  # Short timeout to trigger lock contention
                )
                results.append(result)
            except ModelOperationInProgressError as e:
                errors.append(e)

        # Start two threads trying to load models
        thread1 = threading.Thread(target=load_in_thread)
        thread2 = threading.Thread(target=load_in_thread)

        thread1.start()
        time.sleep(0.02)  # Small delay to ensure thread1 acquires lock first
        thread2.start()

        thread1.join()
        thread2.join()

        # One should succeed, one should fail with ModelOperationInProgressError
        self.assertEqual(1, len(results))
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], ModelOperationInProgressError)

    @patch(
        "inference_orchestrator.services.triton_inference.model_manager.model_manager.get_model_loader"
    )
    def test_load_model_calls_model_load(self, mock_get_loader):
        """Test that _load_model calls the model's load method"""
        mock_triton_client = Mock()
        mock_management_client = Mock()
        mock_model = Mock()
        mock_loader = Mock(return_value=mock_model)
        mock_get_loader.return_value = mock_loader

        model_properties = {"name": "test", "type": "open_clip"}

        result = model_manager._load_model(
            model_properties,
            triton_client=mock_triton_client,
            model_management_client=mock_management_client,
        )

        # Verify model loader was called
        mock_get_loader.assert_called_once_with(model_properties)
        mock_loader.assert_called_once_with(
            model_properties=model_properties,
            model_management_client=mock_management_client,
            triton_client=mock_triton_client,
        )

        # Verify model.load() was called
        mock_model.load.assert_called_once()

        # Verify the model was returned
        self.assertIs(mock_model, result)
