import json
import os
import time
from threading import Lock

import nltk
import torch
from PIL import Image

from marqo import config, version
from marqo import marqo_docs
from marqo.api import exceptions
from marqo.connections import redis_driver
from marqo.s2_inference.constants import PATCH_MODELS
from marqo.s2_inference.processing.image import chunk_image
from marqo.s2_inference.s2_inference import vectorise
# we need to import backend before index_meta_cache to prevent circular import error:
from marqo.tensor_search import constants
from marqo.tensor_search import index_meta_cache, utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo import marqo_docs
import subprocess
import nltk

logger = get_logger(__name__)


def on_start(config: config.Config):
    to_run_on_start = (
        BootstrapVespa(config),
        DownloadStartText(),
        InitializeRedis("localhost", 6379),
        DownloadFinishText(),
        PrintVersion(),
        MarqoWelcome(),
        MarqoPhrase(),
    )

    for thing_to_start in to_run_on_start:
        thing_to_start.run()


class BootstrapVespa:
    """Create the Marqo settings schema on Vespa"""

    def __init__(self, config: config.Config):
        self.config = config

    def run(self):
        try:
            logger.debug('Bootstrapping Vespa')
            created = self.config.index_management.bootstrap_vespa()
            if created:
                logger.debug('Vespa configured successfully')
            else:
                logger.debug('Vespa configuration already exists. Skipping bootstrap')
        except Exception as e:
            logger.error(
                f"Failed to bootstrap vector store. If you are using an external vector store, "
                "ensure that Marqo is configured properly for this. See "
                f"{marqo_docs.configuring_marqo()} for more details. Error: {e}"
            )
            raise e


def _preload_model(model, content, device):
    """
        Calls vectorise for a model once. This will load in the model if it isn't already loaded.
        If `model` is a str, it should be a model name in the registry
        If `model is a dict, it should be an object containing `model_name` and `model_properties`
        Model properties will be passed to vectorise call if object exists
    """
    if isinstance(model, str):
        # For models IN REGISTRY
        _ = vectorise(
            model_name=model,
            content=content,
            device=device
        )
    elif isinstance(model, dict):
        # For models from URL
        """
        TODO: include validation from on start script (model name properties etc)
        _check_model_name(index_settings)
        """
        try:
            _ = vectorise(
                model_name=model["model"],
                model_properties=model["modelProperties"],
                content=content,
                device=device
            )
        except KeyError as e:
            raise exceptions.EnvVarError(
                f"Your custom model {model} is missing either `model` or `model_properties`."
                f"To add a custom model, it must be a dict with keys `model` and `model_properties`. "
                f"See the examples defined in {marqo_docs.configuring_preloaded_models()}"
            ) from e


class InitializeRedis:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def run(self):
        logger.debug('Initializing Redis')
        # Can be turned off with MARQO_ENABLE_THROTTLING = 'FALSE'
        if utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_THROTTLING) == "TRUE":
            redis_driver.init_from_app(self.host, self.port)


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
