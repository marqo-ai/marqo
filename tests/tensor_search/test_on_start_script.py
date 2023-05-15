import json

from tests.marqo_test import MarqoTestCase
from unittest import mock
from marqo.tensor_search import enums, configs
from marqo.tensor_search import on_start_script
from marqo.s2_inference import s2_inference
from marqo import errors


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
            @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
            def run():
                model_caching_script = on_start_script.ModelsForCacheing()
                model_caching_script.run()
                loaded_models = {args[0] for args, kwargs in mock_vectorise.call_args_list}
                assert loaded_models == set(expected)
                return True
            assert run()

    def test_preload_models_malformed(self):
        @mock.patch("os.environ", {enums.EnvVars.MARQO_MODELS_TO_PRELOAD: "[not-good-json"})
        def run():
            try:
                model_caching_script = on_start_script.ModelsForCacheing()
                raise AssertionError
            except errors.EnvVarError as e:
                print(str(e))
                return True
        assert run()
    
    def test_preload_url_models(self):
        environ_expected_models = [
        ]
        for mock_environ, expected in environ_expected_models:
            mock_vectorise = mock.MagicMock()
            @mock.patch("os.environ", mock_environ)
            @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
            def run():
                model_caching_script = on_start_script.ModelsForCacheing()
                model_caching_script.run()
                loaded_models = {args[0] for args, kwargs in mock_vectorise.call_args_list}
                assert loaded_models == set(expected)
                return True
            assert run()
    
    def test_preload_url_models_malformed(self):
        environ_expected_models = [
        ]
        for mock_environ, expected in environ_expected_models:
            mock_vectorise = mock.MagicMock()
            @mock.patch("os.environ", mock_environ)
            @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
            def run():
                model_caching_script = on_start_script.ModelsForCacheing()
                model_caching_script.run()
                loaded_models = {args[0] for args, kwargs in mock_vectorise.call_args_list}
                assert loaded_models == set(expected)
                return True
            assert run()









