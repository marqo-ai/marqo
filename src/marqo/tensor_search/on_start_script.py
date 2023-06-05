import json
import os
import logging
from marqo.tensor_search import enums
from marqo.tensor_search.tensor_search_logging import get_logger
import time
from marqo.tensor_search.enums import EnvVars
# we need to import backend before index_meta_cache to prevent circular import error:
from marqo.tensor_search import backend, index_meta_cache, utils
from marqo import config
from marqo.tensor_search.web import api_utils
from marqo._httprequests import HttpRequests
from marqo import errors
from marqo.tensor_search.throttling.redis_throttle import throttle
from marqo.connections import redis_driver
from functools import wraps

logger = logging.getLogger(__name__)

def on_start(marqo_os_url: str):
    logger.info("Starting the on start script")
    to_run_on_start = (
        PopulateCache(marqo_os_url),
        DownloadStartText(),
        CUDAAvailable(),
        ModelsForCacheing(),
        InitializeRedis("localhost", 6379),  # TODO, have these variable
        DownloadFinishText(),
        MarqoWelcome(),
        MarqoPhrase(),
    )

    for thing_to_start in to_run_on_start:
        thing_to_start.run()


def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"{func.__qualname__} started")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__qualname__} finished")
        elapsed_time = end_time - start_time
        logger.info(f"{func.__qualname__} finished in {elapsed_time} seconds")
        return result

    return wrapper


class PopulateCache:
    """Populates the cache on start"""

    @log_time
    def __init__(self, marqo_os_url: str):
        self.marqo_os_url = marqo_os_url
        pass

    @log_time
    def run(self):
        c = config.Config(api_utils.upconstruct_authorized_url(
            opensearch_url=self.marqo_os_url
        ))
        try:
            index_meta_cache.populate_cache(c)
        except errors.BackendCommunicationError as e:
            raise errors.BackendCommunicationError(
                message="Can't connect to Marqo-os backend. \n"
                        "    Possible causes: \n"
                        "        - If this is an arm64 machine, ensure you are using an external Marqo-os instance \n"
                        "        - If you are using an external Marqo-os instance, check if it is running: "
                        "`curl <YOUR MARQO-OS URL>` \n"
                        "        - Ensure that the OPENSEARCH_URL environment variable defined "
                        "in the `docker run marqo` command points to Marqo-os\n",
                link="https://github.com/marqo-ai/marqo/tree/mainline/src/marqo"
                     "#c-build-and-run-the-marqo-as-a-docker-container-connecting-"
                     "to-marqo-os-which-is-running-on-the-host"
            ) from e
        # the following lines turns off auto create index
        # connection = HttpRequests(c)
        # connection.put(
        #     path="_cluster/settings",
        #     body={
        #         "persistent": {"action.auto_create_index": "false"}
        #     })


class CUDAAvailable:
    """checks the status of cuda
    """
    logger = get_logger('CUDA device summary')

    @log_time
    def __init__(self):

        pass

    @log_time
    def run(self):
        import torch

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
        self.logger.info(f"found devices {device_names}")


class ModelsForCacheing:
    """warms the in-memory model cache by preloading good defaults
    """
    logger = get_logger('ModelsForStartup')

    @log_time
    def __init__(self):
        import torch
        warmed_models = utils.read_env_vars_and_defaults(EnvVars.MARQO_MODELS_TO_PRELOAD)
        if warmed_models is None:
            self.models = []
        elif isinstance(warmed_models, str):
            try:
                self.models = json.loads(warmed_models)
            except json.JSONDecodeError as e:
                raise errors.EnvVarError(
                    f"Could not parse environment variable `{EnvVars.MARQO_MODELS_TO_PRELOAD}`. "
                    f"Please ensure that this a JSON-encoded array of strings. For example:\n"
                    f"""export {EnvVars.MARQO_MODELS_TO_PRELOAD}='["ViT-L/14", "onnx/all_datasets_v4_MiniLM-L6"]'"""
                ) from e
        else:
            self.models = warmed_models
        # TBD to include cross-encoder/ms-marco-TinyBERT-L-2-v2

        self.default_devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']

        self.logger.info(f"pre-loading {self.models} onto devices={self.default_devices}")

    @log_time
    def run(self):
        from marqo.s2_inference.s2_inference import vectorise

        test_string = 'this is a test string'
        N = 10
        messages = []
        for model in self.models:
            for device in self.default_devices:

                # warm it up
                _ = vectorise(model, test_string, device=device)

                t = 0
                for n in range(N):
                    t0 = time.time()
                    _ = vectorise(model, test_string, device=device)
                    t1 = time.time()
                    t += (t1 - t0)
                message = f"{(t) / float((N))} for {model} and {device}"
                messages.append(message)
                self.logger.info(f"{model} {device} run succesfully!")

        for message in messages:
            self.logger.info(message)
        self.logger.info("completed loading models")


class InitializeRedis:
    @log_time
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    @log_time
    def run(self):
        # Can be turned off with MARQO_ENABLE_THROTTLING = 'FALSE'
        if utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_THROTTLING) == "TRUE":
            redis_driver.init_from_app(self.host, self.port)


class DownloadStartText:
    @log_time
    def run(self):
        print('\n')
        print("###########################################################")
        print("###########################################################")
        print("###### STARTING DOWNLOAD OF MARQO ARTEFACTS################")
        print("###########################################################")
        print("###########################################################")
        print('\n')


class DownloadFinishText:
    @log_time
    def run(self):
        print('\n')
        print("###########################################################")
        print("###########################################################")
        print("###### !!COMPLETED SUCCESFULLY!!!          ################")
        print("###########################################################")
        print("###########################################################")
        print('\n')


class MarqoPhrase:
    @log_time
    def run(self):
        message = r"""
     _____                                                   _        __              _                                     
    |_   _|__ _ __  ___  ___  _ __   ___  ___  __ _ _ __ ___| |__    / _| ___  _ __  | |__  _   _ _ __ ___   __ _ _ __  ___ 
      | |/ _ \ '_ \/ __|/ _ \| '__| / __|/ _ \/ _` | '__/ __| '_ \  | |_ / _ \| '__| | '_ \| | | | '_ ` _ \ / _` | '_ \/ __|
      | |  __/ | | \__ \ (_) | |    \__ \  __/ (_| | | | (__| | | | |  _| (_) | |    | | | | |_| | | | | | | (_| | | | \__ \
      |_|\___|_| |_|___/\___/|_|    |___/\___|\__,_|_|  \___|_| |_| |_|  \___/|_|    |_| |_|\__,_|_| |_| |_|\__,_|_| |_|___/

        """

        print(message)


class MarqoWelcome:
    @log_time
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
        print(message)
