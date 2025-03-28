import json
import os
import unittest
from unittest import mock

from marqo.api import exceptions, configs
from marqo.api.exceptions import StartupSanityCheckError
from marqo.core.inference.api import Inference
from marqo.inference.native_inference.remote.server import on_start_script
from marqo.inference.native_inference.remote.server.inference_config import Config
from marqo.tensor_search import enums
from marqo.tensor_search.enums import EnvVars


class TestOnStartScript(unittest.TestCase):

    def setUp(self):
        self.mock_config = mock.MagicMock(spec=Config)
        self.mock_inference = mock.MagicMock(spec=Inference)
        self.mock_config.local_inference = self.mock_inference

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
            @mock.patch("os.environ", mock_environ)
            def run():
                self.mock_inference.reset_mock()
                model_caching_script = on_start_script.CacheModels(self.mock_config)
                model_caching_script.run()
                loaded_models = {args[0].model_config.model_name for args, _ in self.mock_inference.vectorise.call_args_list}
                assert loaded_models == set(expected)
                return True
            assert run()

    def test_preload_models_malformed(self):
        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_MODELS_TO_PRELOAD: "[not-good-json"})
        def run():
            try:
                model_caching_script = on_start_script.CacheModels(self.mock_config)
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
            @mock.patch.dict(os.environ, mock_environ)
            def run():
                self.mock_inference.reset_mock()
                model_caching_script = on_start_script.CacheModels(self.mock_config)
                model_caching_script.run()
                loaded_models = {
                    (
                        args[0].model_config.model_name,
                        args[0].model_config.model_properties["name"],
                        args[0].model_config.model_properties["dimensions"],
                        args[0].model_config.model_properties["type"],
                        args[0].model_config.model_properties["url"],
                    ) for args, _ in self.mock_inference.vectorise.call_args_list
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

        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_MODELS_TO_PRELOAD: json.dumps([open_clip_model_object])})
        def run():
            try:
                model_caching_script = on_start_script.CacheModels(self.mock_config)
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

        @mock.patch.dict(os.environ, {enums.EnvVars.MARQO_MODELS_TO_PRELOAD: json.dumps([open_clip_model_object])})
        def run():
            try:
                model_caching_script = on_start_script.CacheModels(self.mock_config)
                # There should be a KeyError -> EnvVarError when attempting to call vectorise
                model_caching_script.run()
                raise AssertionError
            except exceptions.EnvVarError as e:
                return True
        assert run()
    
    def test_SetEnableVideoGPUAcceleration_none_input_check_fails(self):
        """Test when the env variable is None(not set by the users) and the check fails, the env var is set to 'FALSE'."""
        with mock.patch.dict('marqo.inference.native_inference.remote.server.on_start_script.os.environ',
                             {}), \
        mock.patch('marqo.inference.native_inference.remote.server.on_start_script.SetEnableVideoGPUAcceleration._check_video_gpu_acceleration_availability') as mock_check_gpu_acceleration:
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
        with mock.patch.dict('marqo.inference.native_inference.remote.server.on_start_script.os.environ',
                             {}), \
        mock.patch('marqo.inference.native_inference.remote.server.on_start_script.SetEnableVideoGPUAcceleration._check_video_gpu_acceleration_availability') as mock_check_gpu_acceleration:
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
        with mock.patch.dict('marqo.inference.native_inference.remote.server.on_start_script.os.environ',
                             {EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION: "TRUE"}), \
        mock.patch('marqo.inference.native_inference.remote.server.on_start_script.SetEnableVideoGPUAcceleration._check_video_gpu_acceleration_availability') as mock_check_gpu_acceleration:
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
        with mock.patch("marqo.inference.native_inference.remote.server.on_start_script.nltk.data.find") as mock_find, \
             mock.patch("marqo.inference.native_inference.remote.server.on_start_script.nltk.download") as mock_nltk_download:
                # Mock find to always succeed
                mock_find.side_effect = LookupError()

                checker = on_start_script.CheckNLTKTokenizers()
                with self.assertRaises(StartupSanityCheckError):
                    checker.run()
                mock_nltk_download.assert_any_call("punkt_tab")