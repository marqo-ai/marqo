import json
import os
import subprocess
import time
from threading import Lock
from typing import Dict

import nltk
import torch
from PIL import Image

from marqo import marqo_docs
from marqo import version
from marqo.api import exceptions  # TODO EnvVarError and StartupSanityCheckError will need to be replicated here
from marqo.core.inference.api import ModelConfig, InferenceRequest, Modality, TextPreprocessingConfig
from marqo.exceptions import InvalidArgumentError
from marqo.inference.native_inference.remote.server.inference_config import Config
from marqo.logging import get_logger

# TODO remove deps of s2_inference
from marqo.s2_inference import s2_inference
from marqo.s2_inference.constants import PATCH_MODELS
from marqo.s2_inference.errors import UnknownModelError, InvalidModelPropertiesError
from marqo.s2_inference.processing.image import chunk_image

# TODO remove these deps
from marqo.tensor_search import constants
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars

logger = get_logger(__name__)


def on_start(config: Config):
    to_run_on_start = (
        DownloadStartText(),
        CUDAAvailable(),
        SetEnableVideoGPUAcceleration(),
        CheckNLTKTokenizers(),
        CacheModels(config),
        # CachePatchModels(),  # TODO patch model can be deprecated, we comment it out for now
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

        warmed_models = utils.read_env_vars_and_defaults(EnvVars.MARQO_MODELS_TO_PRELOAD)
        if warmed_models is None:
            self.models = []
        elif isinstance(warmed_models, str):
            try:
                self.models = json.loads(warmed_models)
            except json.JSONDecodeError as e:
                # TODO: Change error message to match new format
                raise exceptions.EnvVarError(
                    f"Could not parse environment variable `{EnvVars.MARQO_MODELS_TO_PRELOAD}`. "
                    f"Please ensure that this a JSON-encoded array of strings or dicts. For example:\n"
                    f"""export {EnvVars.MARQO_MODELS_TO_PRELOAD}='["hf/e5-base-v2", "open_clip/ViT-B-32/laion2b_s34b_b79k"]'"""
                    f"To add a custom model, it must be a dict with keys `model` and `model_properties` "
                    f"as defined in {marqo_docs.bring_your_own_model()}"
                ) from e
        else:
            self.models = warmed_models
        # TBD to include cross-encoder/ms-marco-TinyBERT-L-2-v2

        # TODO [Refactoring device logic] use device info gathered from device manager
        self.default_devices = ['cpu'] if not torch.cuda.is_available() else ['cuda', 'cpu']

        self.logger.info(f"pre-loading {self.models} onto devices={self.default_devices}")

    def run(self):
        test_string = 'this is a test string'
        N = 10
        messages = []
        for model in self.models:
            # Skip preloading of models that can't be preloaded (eg. no_model)
            if isinstance(model, str):
                model_name = model
            elif isinstance(model, dict):
                try:
                    model_name = model["model"]
                except KeyError as e:
                    raise exceptions.EnvVarError(
                        f"Your custom model {model} is missing 'model' key."
                        f"To add a custom model, it must be a dict with keys 'model' and 'model_properties' "
                        f"as defined in '{marqo_docs.configuring_preloaded_models()}'"
                    ) from e
            else:
                continue

            if model_name in constants.MODELS_TO_SKIP_PRELOADING:
                self.logger.info(
                    f"Skipping preloading of '{model_name}' because the model does not require preloading.")
                continue
            for device in self.default_devices:
                self.logger.debug(f"Loading model: {model} on device: {device}")

                # warm it up
                _ = self._preload_model(model=model, content=test_string, device=device)

                t = 0
                for n in range(N):
                    t0 = time.time()
                    _ = self._preload_model(model=model, content=test_string, device=device)
                    t1 = time.time()
                    t += (t1 - t0)
                message = f"{(t) / float((N))} for {model} and {device}"
                messages.append(message)
                self.logger.debug(f"{model} {device} vectorise run {N} times.")
                self.logger.info(f"{model} {device} run succesfully!")

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
            try:
                model_config = ModelConfig(
                    model_name=model["model"],
                    model_properties=model["modelProperties"],
                )
            except KeyError as e:
                raise exceptions.EnvVarError(
                    f"Your custom model {model} is missing either `model` or `model_properties`."
                    f"To add a custom model, it must be a dict with keys `model` and `model_properties`. "
                    f"See the examples defined in {marqo_docs.configuring_preloaded_models()}"
                ) from e

        _ = self.config.local_inference.vectorise(InferenceRequest(
            modality=Modality.TEXT,
            contents=[content],
            model_config=model_config,
            preprocessing_config=TextPreprocessingConfig(),
            device=device
        ))

    def _load_model_properties_from_model_registry(self, model_name: str) -> Dict[str, str]:
        try:
            # TODO expose this via model manager class !!!
            return s2_inference.get_model_properties_from_registry(model_name)
        except UnknownModelError:
            raise InvalidArgumentError(
                f'Could not find model properties for model={model_name}. '
                f'Please check that the model name is correct. '
                f'Please provide model_properties if the model is a custom model and is not supported by default')
        except InvalidModelPropertiesError as e:
            raise InvalidArgumentError(
                f'Invalid model properties for model={model_name}. Reason: {e}.'
            )


class CachePatchModels:
    """Prewarm patch models"""

    logger = get_logger('CachePatchModels')
    lock = Lock()

    def __init__(self):
        # TODO MARQO_PATCH_MODELS_TO_PRELOAD does not have default value in config.py
        warmed_models = utils.read_env_vars_and_defaults(EnvVars.MARQO_PATCH_MODELS_TO_PRELOAD)
        if warmed_models is None:
            self.models = []
        elif isinstance(warmed_models, str):
            try:
                self.models = json.loads(warmed_models)
            except json.JSONDecodeError as e:
                raise exceptions.EnvVarError(
                    f"Could not parse environment variable `{EnvVars.MARQO_PATCH_MODELS_TO_PRELOAD}`. "
                    f"Please ensure that this is a JSON-encoded list of strings."
                ) from e
        elif isinstance(warmed_models, list):
            self.models = warmed_models
        else:
            raise exceptions.EnvVarError(
                f"Environment variable `{EnvVars.MARQO_PATCH_MODELS_TO_PRELOAD}` should be a list of strings."
            )

        for model in self.models:
            if model not in PATCH_MODELS:
                raise exceptions.EnvVarError(
                    f"Invalid patch model: {model}. Please ensure that this is a valid patch model."
                )

        # TODO [Refactoring device logic] use device info gathered from device manager
        self.default_devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']

    def run(self):
        N = 10
        messages = []
        test_image = torch.zeros((3, 224, 224))  # Dummy image tensor
        test_image_pil = Image.fromarray(test_image.numpy().astype('uint8').transpose(1, 2, 0))  # Convert to PIL image
        for model in self.models:
            for device in self.default_devices:
                self.logger.debug(f"Prewarming model: {model} on device: {device}")
                with self.lock:
                    try:
                        # Warm it up
                        chunks = chunk_image(test_image_pil, device=device, method=model)

                        t = 0
                        for n in range(N):
                            t0 = time.time()
                            chunks = chunk_image(test_image_pil, device=device, method=model)
                            t1 = time.time()
                            t += (t1 - t0)
                        message = f"{(t) / float((N))} for {model} and {device}"
                        messages.append(message)
                        self.logger.debug(f"{model} {device} ran chunking {N} times.")
                        self.logger.info(f"{model} {device} chunking run succesfully!")

                    except Exception as e:
                        self.logger.error(f"Failed to prewarm model: {model} on device: {device}. Error: {e}")

                self.logger.info(f"Prewarmed model: {model} on device: {device}")

        for message in messages:
            self.logger.info(message)
        self.logger.info("completed prewarming patch models")


class SetEnableVideoGPUAcceleration:
    logger = get_logger('SetVideoProcessingDevice')

    def run(self):
        """This method will set the env var MARQO_ENABLE_VIDEO_GPU_ACCELERATION to TRUE or FALSE."""
        env_value = utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION)
        if env_value is None:
            try:
                self._check_video_gpu_acceleration_availability()
                os.environ[EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION] = "TRUE"
            except exceptions.StartupSanityCheckError as e:
                self.logger.debug(f"Failed to use GPU acceleration for video processing. We will disable it. "
                                  f"Original error message: {e}")
                os.environ[EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION] = "FALSE"
                logger.info(f"Video processing with GPU acceleration is disabled")
        elif env_value == "TRUE":
            self._check_video_gpu_acceleration_availability()
            logger.info(f"Video processing with GPU acceleration is enabled")
        elif env_value == "FALSE":
            logger.info(f"Video processing with GPU acceleration is disabled")
            pass
        else:
            raise exceptions.EnvVarError(
                f"Invalid value for {EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION}. "
                f"Please set it to either 'TRUE' or 'FALSE'."
            )

    def _check_video_gpu_acceleration_availability(self):
        """Check if the required dependencies are available for video processing with GPU acceleration for ffmpeg.

        Raises:
            exceptions.StartupSanityCheckError: If the required dependencies are not available.
        """
        ffmpeg_command_gpu_check = [
            'ffmpeg',
            '-v', 'error',  # Suppress output
            '-hwaccel', 'cuda',  # Use CUDA for hardware acceleration
            '-f', 'lavfi',  # Input format is a lavfi (FFmpeg's built-in filter)
            '-i', 'nullsrc=s=200x100',  # Generate a blank video source of 200x100 resolution
            '-vframes', '1',  # Process only 1 frame
            '-c:v', 'h264_nvenc',  # Use NVENC encoder
            '-f', 'null',  # Output to null (discard the output)
            '-'  # Output to stdout (discarded)
        ]
        try:
            _ = subprocess.run(
                ffmpeg_command_gpu_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                text=True, timeout=10
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise exceptions.StartupSanityCheckError(
                f"Failed to use GPU acceleration for video processing. "
                f"Ensure that your system has the required dependencies installed. "
                f"You can set 'MARQO_ENABLE_VIDEO_GPU_ACCELERATION=FALSE' to disable GPU acceleration. "
                f"Check {marqo_docs.configuring_marqo()} for more information. "
                f"Original error message: {e.stderr}"
            ) from e
        except (ValueError, OSError) as e:
            raise exceptions.StartupSanityCheckError(
                f"Marqo failed to run the ffmpeg sanity check. Your ffmepeg installation might be broken. "
                f"Original error: {e}"
            ) from e


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
            raise exceptions.StartupSanityCheckError(
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
        print(f"Version: {version.__version__}")


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
