import time
from typing import Dict, Union

from inference_orchestrator.services.triton_inference.embedding_models.marqo_model_registry import (
    get_model_properties,
)
from inference_orchestrator.version import get_version

from .config import Config
from .core.logging import get_logger
from .core.settings import get_settings
from .schemas.api import (
    EmbeddingModelConfig,
    InferenceRequest,
    Modality,
    TextPreprocessingConfig,
)

settings = get_settings()
logger = get_logger(__name__)


def on_start(config: Config):
    to_run_on_start = (
        CacheModels(config),
        PrintVersion(),
    )

    for thing_to_start in to_run_on_start:
        thing_to_start.run()


class CacheModels:
    """warms the in-memory model cache by preloading good defaults"""

    logger = get_logger("ModelsForStartup")

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        test_string = "this is a test string"
        N = 10
        messages = []
        for model in settings.marqo_models_to_preload:
            if isinstance(model, dict):
                model_name = model["model"]
            else:
                model_name = model

            self.logger.debug(f"Loading model: {model_name}")

            # warm it up
            _ = self._preload_model(model=model, content=test_string)

            t = 0
            for n in range(N):
                t0 = time.time()
                _ = self._preload_model(model=model, content=test_string)
                t1 = time.time()
                t += t1 - t0
            message = f"{(t) / float((N))} for {model} over {N} runs"
            messages.append(message)
            self.logger.info(f"{model} warm-up successfully!")

        for message in messages:
            self.logger.info(message)
        self.logger.info("completed loading models")

    def _preload_model(self, model: Union[str, dict], content: str):
        """
        Calls vectorise for a model once. This will load in the model if it isn't already loaded.
        If `model` is a str, it should be a model name in the registry
        If `model is a dict, it should be an object containing `model_name` and `model_properties`
        Model properties will be passed to vectorise call if object exists
        """
        model_config = None
        if isinstance(model, str):
            # For models IN REGISTRY
            model_config = EmbeddingModelConfig(
                model_name=model,
                model_properties=self._load_model_properties_from_model_registry(model),
            )
        elif isinstance(model, dict):
            # For models from URL
            """
            TODO: include validation from on start script (model name properties etc)
            _check_model_name(index_settings)
            """
            model_config = EmbeddingModelConfig(
                model_name=model["model"],
                model_properties=model["modelProperties"],
            )

        _ = self.config.inference.vectorise(
            InferenceRequest(
                modality=Modality.TEXT,
                contents=[content],
                embedding_model_config=model_config,
                preprocessing_config=TextPreprocessingConfig(),
            )
        )

    def _load_model_properties_from_model_registry(
        self, model_name: str
    ) -> Dict[str, str]:
        return get_model_properties(model_name)


class PrintVersion:
    def run(self):
        print(f"Version: {get_version()}")
