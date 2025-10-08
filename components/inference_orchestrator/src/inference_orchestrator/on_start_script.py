import time
from typing import Dict

import nltk
import torch

from inference_orchestrator.services.triton_inference.embedding_models.marqo_model_regiestry import \
    get_model_properties
from inference_orchestrator.version import get_version
from .config import Config
from .core.logging import get_logger
from .core.settings import get_settings
from .errors.common_errors import StartupSanityCheckError
from .schemas.api import ModelConfig, InferenceRequest, Modality, TextPreprocessingConfig

settings = get_settings()
logger = get_logger(__name__)


def on_start(config: Config):
    to_run_on_start = (
        DownloadStartText(),
        CUDAAvailable(),
        CheckNLTKTokenizers(),
        CacheModels(config),
        DownloadFinishText(),
        PrintVersion(),

        # TODO do we still need banners? or a different banner?
        MarqoWelcome(),
        MarqoPhrase(),
    )

    for thing_to_start in to_run_on_start:
        thing_to_start.run()


class CUDAAvailable:
    # TODO [Refactoring device logic] move this logic to device manager
    """checks the status of cuda
    """
    logger = get_logger('CUDA device summary')

    def run(self):
        def id_to_device(id):
            if id < 0:
                return ['cpu']
            return [torch.cuda.get_device_name(id)]

        device_count = 0 if not torch.cuda.is_available() else torch.cuda.device_count()

        # use -1 for cpu
        device_ids = [-1]
        device_ids += list(range(device_count))

        device_names = []
        for device_id in device_ids:
            device_names.append({'id': device_id, 'name': id_to_device(device_id)})

        self.logger.info(f"Found devices {device_names}")


class CacheModels:
    """warms the in-memory model cache by preloading good defaults
    """
    logger = get_logger('ModelsForStartup')

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        test_string = 'this is a test string'
        N = 10
        messages = []
        for model in settings.marqo_models_to_preload:
            # Skip preloading of models that can't be preloaded (eg. no_model)
            if isinstance(model, str):
                model_name = model
            elif isinstance(model, dict):
                model_name = model["model"]

            if model_name in {"no_model"}:
                self.logger.info(
                    f"Skipping preloading of '{model_name}' because the model does not require preloading.")
                continue

            self.logger.debug(f"Loading model: {model}")

            # warm it up
            _ = self._preload_model(model=model, content=test_string)

            t = 0
            for n in range(N):
                t0 = time.time()
                _ = self._preload_model(model=model, content=test_string)
                t1 = time.time()
                t += (t1 - t0)
            message = f"{(t) / float((N))} for {model} over {N} runs"
            messages.append(message)
            self.logger.info(f"{model} warm-up successfully!")

        for message in messages:
            self.logger.info(message)
        self.logger.info("completed loading models")

    def _preload_model(self, model, content, device):
        """
            Calls vectorise for a model once. This will load in the model if it isn't already loaded.
            If `model` is a str, it should be a model name in the registry
            If `model is a dict, it should be an object containing `model_name` and `model_properties`
            Model properties will be passed to vectorise call if object exists
        """
        model_config = None
        if isinstance(model, str):
            # For models IN REGISTRY
            model_config = ModelConfig(
                model_name=model,
                model_properties=self._load_model_properties_from_model_registry(model)
            )
        elif isinstance(model, dict):
            # For models from URL
            """
            TODO: include validation from on start script (model name properties etc)
            _check_model_name(index_settings)
            """
            model_config = ModelConfig(
                model_name=model["model"],
                model_properties=model["modelProperties"],
            )

        _ = self.config.local_inference.vectorise(InferenceRequest(
            modality=Modality.TEXT,
            contents=[content],
            model_config=model_config,
            preprocessing_config=TextPreprocessingConfig(),
            device=device
        ))

    def _load_model_properties_from_model_registry(self, model_name: str) -> Dict[str, str]:
        return get_model_properties(model_name)


class CheckNLTKTokenizers:
    """Check if NLTK tokenizers are available, if not, download them.

    NLTK tokenizers are included in the base-image, we do a sanity check to ensure they are available.
    """

    def run(self):
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logger.info("NLTK punkt_tab tokenizer not found. Downloading...")
            nltk.download("punkt_tab")

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError as e:
            raise StartupSanityCheckError(
                f"Marqo failed to download and download NLTK tokenizers. Original error: {e}"
            ) from e


class DownloadStartText:

    def run(self):
        print('\n')
        print("###########################################################")
        print("###########################################################")
        print("###### STARTING DOWNLOAD OF MARQO ARTEFACTS################")
        print("###########################################################")
        print("###########################################################")
        print('\n', flush=True)


class DownloadFinishText:

    def run(self):
        print('\n')
        print("###########################################################")
        print("###########################################################")
        print("###### !!COMPLETED SUCCESSFULLY!!!         ################")
        print("###########################################################")
        print("###########################################################")
        print('\n', flush=True)


class PrintVersion:
    def run(self):
        print(f"Version: {get_version()}")


class MarqoPhrase:

    def run(self):
        message = r"""
     _____                                                   _        __              _                                     
    |_   _|__ _ __  ___  ___  _ __   ___  ___  __ _ _ __ ___| |__    / _| ___  _ __  | |__  _   _ _ __ ___   __ _ _ __  ___ 
      | |/ _ \ '_ \/ __|/ _ \| '__| / __|/ _ \/ _` | '__/ __| '_ \  | |_ / _ \| '__| | '_ \| | | | '_ ` _ \ / _` | '_ \/ __|
      | |  __/ | | \__ \ (_) | |    \__ \  __/ (_| | | | (__| | | | |  _| (_) | |    | | | | |_| | | | | | | (_| | | | \__ \
      |_|\___|_| |_|___/\___/|_|    |___/\___|\__,_|_|  \___|_| |_| |_|  \___/|_|    |_| |_|\__,_|_| |_| |_|\__,_|_| |_|___/

        """

        print(message, flush=True)


class MarqoWelcome:

    def run(self):
        message = r"""   
     __    __    ___  _        __   ___   ___ ___    ___      ______   ___       ___ ___   ____  ____   ___    ___   __ 
    |  |__|  |  /  _]| |      /  ] /   \ |   |   |  /  _]    |      | /   \     |   |   | /    ||    \ /   \  /   \ |  |
    |  |  |  | /  [_ | |     /  / |     || _   _ | /  [_     |      ||     |    | _   _ ||  o  ||  D  )     ||     ||  |
    |  |  |  ||    _]| |___ /  /  |  O  ||  \_/  ||    _]    |_|  |_||  O  |    |  \_/  ||     ||    /|  Q  ||  O  ||__|
    |  `  '  ||   [_ |     /   \_ |     ||   |   ||   [_       |  |  |     |    |   |   ||  _  ||    \|     ||     | __ 
     \      / |     ||     \     ||     ||   |   ||     |      |  |  |     |    |   |   ||  |  ||  .  \     ||     ||  |
      \_/\_/  |_____||_____|\____| \___/ |___|___||_____|      |__|   \___/     |___|___||__|__||__|\_|\__,_| \___/ |__|

        """
        print(message, flush=True)
