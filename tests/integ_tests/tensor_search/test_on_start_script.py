import json
import os
from unittest import mock

from marqo.api import exceptions, configs
from marqo.api.exceptions import StartupSanityCheckError
from marqo.tensor_search import enums
from marqo.tensor_search import on_start_script
from marqo.tensor_search.enums import EnvVars
from tests.marqo_test import MarqoTestCase


class TestOnStartScript(MarqoTestCase):

    def test_preload_registry_models(self):
        environ_expected_models = [
            ({enums.EnvVars.MARQO_MODELS_TO_PRELOAD: []}, []),
            ({enums.EnvVars.MARQO_MODELS_TO_PRELOAD: ""}, []),
            (dict(), configs.default_env_vars()[enums.EnvVars.MARQO_MODELS_TO_PRELOAD]),
            ({enums.EnvVars.MARQO_MODELS_TO_PRELOAD: ["sentence-transformers/stsb-xlm-r-multilingual"]},
             ["sentence-transformers/stsb-xlm-r-multilingual"]),
            ({enums.EnvVars.MARQO_MODELS_TO_PRELOAD: json.dumps(["sentence-transformers/stsb-xlm-r-multilingual"])},
             ["sentence-transformers/stsb-xlm-r-multilingual"]),
            ({enums.EnvVars.MARQO_MODELS_TO_PRELOAD: ["sentence-transformers/stsb-xlm-r-multilingual", "hf/all_datasets_v3_mpnet-base"]},
             ["sentence-transformers/stsb-xlm-r-multilingual", "hf/all_datasets_v3_mpnet-base"]),
            ({enums.EnvVars.MARQO_MODELS_TO_PRELOAD: json.dumps(
                ["sentence-transformers/stsb-xlm-r-multilingual", "hf/all_datasets_v3_mpnet-base"])},
             ["sentence-transformers/stsb-xlm-r-multilingual", "hf/all_datasets_v3_mpnet-base"]),
        ]
        for mock_environ, expected in environ_expected_models:
            mock_vectorise = mock.MagicMock()
            @mock.patch("os.environ", mock_environ)
            @mock.patch("marqo.tensor_search.on_start_script.vectorise", mock_vectorise)
            def run():
                model_caching_script = on_start_script.CacheModels()
                model_caching_script.run()
                loaded_models = {kwargs["model_name"] for args, kwargs in mock_vectorise.call_args_list}
                assert loaded_models == set(expected)
                return True
            assert run()

    def test_preload_models_malformed(self):
        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_MODELS_TO_PRELOAD: "[not-good-json"})
        def run():
            try:
                model_caching_script = on_start_script.CacheModels()
                raise AssertionError
            except exceptions.EnvVarError as e:
                print(str(e))
                return True
        assert run()
    
    def test_preload_url_models(self):
        clip_model_object = {
            "model": "generic-clip-test-model-2",
            "modelProperties": {
                "name": "ViT-B/32",
                "dimensions": 512,
                "type": "clip",
                "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
            }
        }

        clip_model_expected = (
            "generic-clip-test-model-2",
            "ViT-B/32", 
            512, 
            "clip", 
            "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
        )

        open_clip_model_object = {
            "model": "random-open-clip-1",
            "modelProperties": {
                "name": "ViT-B-32-quickgelu",
                "dimensions": 512,
                "type": "open_clip",
                "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
            }
        }

        # must be an immutable datatype
        open_clip_model_expected = (
            "random-open-clip-1", 
            "ViT-B-32-quickgelu", 
            512, 
            "open_clip", 
            "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
        )
        
        # So far has clip and open clip tests
        environ_expected_models = [
            ({enums.EnvVars.MARQO_MODELS_TO_PRELOAD: json.dumps([clip_model_object, open_clip_model_object])}, [clip_model_expected, open_clip_model_expected])
        ]
        for mock_environ, expected in environ_expected_models:
            mock_vectorise = mock.MagicMock()
            @mock.patch.dict(os.environ, mock_environ)
            @mock.patch("marqo.tensor_search.on_start_script.vectorise", mock_vectorise)
            def run():
                model_caching_script = on_start_script.CacheModels()
                model_caching_script.run()
                loaded_models = {
                    (
                        kwargs["model_name"],
                        kwargs["model_properties"]["name"],
                        kwargs["model_properties"]["dimensions"],
                        kwargs["model_properties"]["type"],
                        kwargs["model_properties"]["url"]
                    )
                    for args, kwargs in mock_vectorise.call_args_list
                }
                assert loaded_models == set(expected)
                return True
            assert run()
    
    def test_preload_url_missing_model(self):
        open_clip_model_object = {
            "model_properties": {
                "name": "ViT-B-32-quickgelu",
                "dimensions": 512,
                "type": "open_clip",
                "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
            }
        }
        mock_vectorise = mock.MagicMock()
        @mock.patch("marqo.tensor_search.on_start_script.vectorise", mock_vectorise)
        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_MODELS_TO_PRELOAD: json.dumps([open_clip_model_object])})
        def run():
            try:
                model_caching_script = on_start_script.CacheModels()
                # There should be a KeyError -> EnvVarError when attempting to call vectorise
                model_caching_script.run()
                raise AssertionError
            except exceptions.EnvVarError as e:
                return True
        assert run()
    
    def test_preload_url_missing_model_properties(self):
        open_clip_model_object = {
            "model": "random-open-clip-1"
        }
        mock_vectorise = mock.MagicMock()
        @mock.patch("marqo.tensor_search.on_start_script.vectorise", mock_vectorise)
        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_MODELS_TO_PRELOAD: json.dumps([open_clip_model_object])})
        def run():
            try:
                model_caching_script = on_start_script.CacheModels()
                # There should be a KeyError -> EnvVarError when attempting to call vectorise
                model_caching_script.run()
                raise AssertionError
            except exceptions.EnvVarError as e:
                return True
        assert run()
    
    # TODO: test bad/no names/URLS in end-to-end tests, as this logic is done in vectorise call

    def test_set_best_available_device(self):
        """
        Makes sure best available device corresponds to whether or not cuda is available
        """
        test_cases = [
            (True, "cuda"),
            (False, "cpu")
        ]
        mock_cuda_is_available = mock.MagicMock()

        for given_cuda_available, expected_best_device in test_cases:
            mock_cuda_is_available.return_value = given_cuda_available
            @mock.patch("torch.cuda.is_available", mock_cuda_is_available)
            def run():
                # make sure env var is empty first
                os.environ.pop("MARQO_BEST_AVAILABLE_DEVICE", None)
                assert "MARQO_BEST_AVAILABLE_DEVICE" not in os.environ

                set_best_available_device_script = on_start_script.SetBestAvailableDevice()
                set_best_available_device_script.run()
                assert os.environ["MARQO_BEST_AVAILABLE_DEVICE"] == expected_best_device
                return True
            
            assert run()

    def test_preload_no_model(self):
        no_model_object = {
            "model": "no_model",
            "model_properties": {
                "dimensions": 123,
                "type": "no_model"
            }
        }
        mock_preload = mock.MagicMock()

        @mock.patch("marqo.tensor_search.on_start_script._preload_model", mock_preload)
        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_MODELS_TO_PRELOAD: json.dumps([no_model_object])})
        def run():
            model_caching_script = on_start_script.CacheModels()
            model_caching_script.run()
            mock_preload.assert_not_called()  # The preloading function should never be called for no_model

        run()
    
    def test_preload_patch_models(self):
        valid_patch_models = ['simple', 'overlap', 'fastercnn', 'frcnn', 'marqo-yolo', 'yolox', 'dino-v1', 'dino-v2', 'dino/v1', 'dino/v2']
        mock_chunk_image = mock.MagicMock()

        for model in valid_patch_models:
            mock_environ = {enums.EnvVars.MARQO_PATCH_MODELS_TO_PRELOAD: json.dumps([model])}
            expected = [model]

            @mock.patch.dict(os.environ, mock_environ)
            @mock.patch("marqo.tensor_search.on_start_script.chunk_image", mock_chunk_image)
            def run():
                mock_chunk_image.reset_mock()
                model_caching_script = on_start_script.CachePatchModels()
                model_caching_script.run()
                loaded_models = {kwargs["method"] for args, kwargs in mock_chunk_image.call_args_list}
                assert loaded_models == set(expected)
                return True

            assert run()

    def test_preload_invalid_and_valid_patchmodels(self):
        invalid_patch_models = ["invalid_model", "simple"]
        mock_chunk_image = mock.MagicMock()

        # Test with invalid patch models
        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_PATCH_MODELS_TO_PRELOAD: json.dumps(invalid_patch_models)})
        @mock.patch("marqo.tensor_search.on_start_script.chunk_image", mock_chunk_image)
        def run_invalid():
            try:
                model_caching_script = on_start_script.CachePatchModels()
                model_caching_script.run()
                raise AssertionError("Expected EnvVarError not raised")
            except exceptions.EnvVarError as e:
                print(str(e))
                return True

        assert run_invalid()

    def test_preload_empty_patchmodel(self):
        empty_patch_models = []
        mock_chunk_image = mock.MagicMock()

        # Test with empty patch models
        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_PATCH_MODELS_TO_PRELOAD: json.dumps(empty_patch_models)})
        @mock.patch("marqo.tensor_search.on_start_script.chunk_image", mock_chunk_image)
        def run_empty():
            try:
                model_caching_script = on_start_script.CachePatchModels()
                model_caching_script.run()
                # No exception should be raised for an empty list
                return True
            except exceptions.EnvVarError as e:
                print(str(e))
                return False

        assert run_empty()

    def test_concurrent_model_loading(self):
        valid_patch_models = ['dino-v1', 'dino-v2']
        mock_chunk_image = mock.MagicMock()

        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_PATCH_MODELS_TO_PRELOAD: json.dumps(valid_patch_models)})
        @mock.patch("marqo.tensor_search.on_start_script.chunk_image", mock_chunk_image)
        def run():
            model_caching_script = on_start_script.CachePatchModels()
            model_caching_script.run()
            loaded_models = {kwargs["method"] for args, kwargs in mock_chunk_image.call_args_list}
            assert loaded_models == set(valid_patch_models)
            return True

        assert run()

    @mock.patch("marqo.config.Config")
    def test_boostrap_failure_should_raise_error(self, mock_config):
        mock_config.index_management.bootstrap_vespa.side_effect = Exception('some error')

        with self.assertRaises(Exception) as context:
            on_start_script.on_start(mock_config)

        self.assertTrue('some error' in str(context.exception))


    def test_SetEnableVideoGPUAcceleration_none_input_check_fails(self):
        """Test when the env variable is None(not set by the users) and the check fails, the env var is set to 'FALSE'."""
        with mock.patch.dict('marqo.tensor_search.on_start_script.os.environ',
                             {}), \
        mock.patch('marqo.tensor_search.on_start_script.SetEnableVideoGPUAcceleration._check_video_gpu_acceleration_availability') as mock_check_gpu_acceleration:
            mock_check_gpu_acceleration.side_effect = exceptions.StartupSanityCheckError('GPU not available')

            # Create instance of the class
            obj = on_start_script.SetEnableVideoGPUAcceleration()

            # Run the method
            obj.run()

            # Assertions
            mock_check_gpu_acceleration.assert_called_once()
            self.assertEqual("FALSE", os.environ[EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION])

    def test_SetEnableVideoGPUAcceleration_none_input_check_pass(self):
        """Test when the env variable is None(not set by the users) and the check pass, the env var is set to 'TRUE'."""
        with mock.patch.dict('marqo.tensor_search.on_start_script.os.environ',
                             {}), \
        mock.patch('marqo.tensor_search.on_start_script.SetEnableVideoGPUAcceleration._check_video_gpu_acceleration_availability') as mock_check_gpu_acceleration:
            mock_check_gpu_acceleration.return_value = None

            # Create instance of the class
            obj = on_start_script.SetEnableVideoGPUAcceleration()

            # Run the method
            obj.run()

            # Assertions
            mock_check_gpu_acceleration.assert_called_once()
            self.assertEqual("TRUE", os.environ[EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION])

    def test_SetEnableVideoGPUAccelerationTrueButCheckFails(self):
        """Test when the env variable is TRUE and the check fails, an error raised."""
        with mock.patch.dict('marqo.tensor_search.on_start_script.os.environ',
                             {EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION: "TRUE"}), \
        mock.patch('marqo.tensor_search.on_start_script.SetEnableVideoGPUAcceleration._check_video_gpu_acceleration_availability') as mock_check_gpu_acceleration:
            mock_check_gpu_acceleration.side_effect = exceptions.StartupSanityCheckError('GPU not available')

            # Create instance of the class
            obj = on_start_script.SetEnableVideoGPUAcceleration()

            # Run the method
            with self.assertRaises(exceptions.StartupSanityCheckError):
                obj.run()

            # Assertions
            mock_check_gpu_acceleration.assert_called_once()

    def test_missing_punkt_downloaded(self):
        """A test to ensure that the script will attempt to download the punkt_tab
        tokenizer if it is not found"""
        with mock.patch("marqo.tensor_search.on_start_script.nltk.data.find") as mock_find, \
            mock.patch("marqo.tensor_search.on_start_script.nltk.download") as mock_nltk_download:
                # Mock find to always succeed
                mock_find.side_effect = LookupError()

                checker = on_start_script.CheckNLTKTokenizers()
                with self.assertRaises(StartupSanityCheckError):
                    checker.run()
                mock_nltk_download.assert_any_call("punkt_tab")