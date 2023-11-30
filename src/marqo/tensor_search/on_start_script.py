import json
import os
from marqo.tensor_search import enums
from marqo.tensor_search.tensor_search_logging import get_logger
import time
from marqo.tensor_search.enums import EnvVars
# we need to import backend before index_meta_cache to prevent circular import error:
from marqo.tensor_search import backend, index_meta_cache, utils
from marqo import config
from marqo.tensor_search.web import api_utils
from marqo import errors
from marqo.tensor_search.throttling.redis_throttle import throttle
from marqo.connections import redis_driver
from marqo.s2_inference.s2_inference import vectorise
import torch


def on_start(marqo_os_url: str):
        
    to_run_on_start = (
                        PopulateCache(marqo_os_url),
                        DownloadStartText(),
                        CUDAAvailable(), 
                        SetBestAvailableDevice(),
                        ModelsForCacheing(),
                        InitializeRedis("localhost", 6379),    # TODO, have these variable
                        DownloadFinishText(),
                        MarqoWelcome(),
                        MarqoPhrase(),
                        )

    for thing_to_start in to_run_on_start:
        thing_to_start.run()


class PopulateCache:
    """Populates the cache on start"""

    def __init__(self, marqo_os_url: str):
        self.marqo_os_url = marqo_os_url
        pass

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
    logger = get_logger('DeviceSummary')

    def __init__(self):
        
        pass

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
            device_names.append( {'id':device_id, 'name':id_to_device(device_id)})

        self.logger.info(f"found devices {device_names}")


class SetBestAvailableDevice:

    """sets the MARQO_BEST_AVAILABLE_DEVICE env var
    """
    logger = get_logger('SetBestAvailableDevice')

    def __init__(self):
        pass

    def run(self):
        """
            This is set once at startup time. We assume it will NOT change,
            if it does, health check should throw a warning.
        """
        if torch.cuda.is_available():
            os.environ[EnvVars.MARQO_BEST_AVAILABLE_DEVICE] = "cuda"
        else:
            os.environ[EnvVars.MARQO_BEST_AVAILABLE_DEVICE] = "cpu"
        
        self.logger.info(f"Best available device set to: {os.environ[EnvVars.MARQO_BEST_AVAILABLE_DEVICE]}")


class ModelsForCacheing:
    """warms the in-memory model cache by preloading good defaults
    """
    logger = get_logger('ModelsForStartup')

    def __init__(self):
        warmed_models = utils.read_env_vars_and_defaults(EnvVars.MARQO_MODELS_TO_PRELOAD)
        if warmed_models is None:
            self.models = []
        elif isinstance(warmed_models, str):
            try:
                self.models = json.loads(warmed_models)
            except json.JSONDecodeError as e:
                # TODO: Change error message to match new format
                raise errors.EnvVarError(
                    f"Could not parse environment variable `{EnvVars.MARQO_MODELS_TO_PRELOAD}`. "
                    f"Please ensure that this a JSON-encoded array of strings or dicts. For example:\n"
                    f"""export {EnvVars.MARQO_MODELS_TO_PRELOAD}='["ViT-L/14", "onnx/all_datasets_v4_MiniLM-L6"]'"""
                    f"""To add a custom model, it must be a dict with keys `model` and `model_properties` as defined in `https://docs.marqo.ai/0.0.20/Models-Reference/bring_your_own_model/`"""
                ) from e
        else:
            self.models = warmed_models
        # TBD to include cross-encoder/ms-marco-TinyBERT-L-2-v2

        self.default_devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']

        self.logger.info(f"pre-loading {self.models} onto devices={self.default_devices}")

    def run(self):
        test_string = 'this is a test string'
        N = 10
        messages = []
        for model in self.models:
            for device in self.default_devices:
                self.logger.debug(f"Beginning loading for model: {model} on device: {device}")
                
                # warm it up
                _ = _preload_model(model=model, content=test_string, device=device)

                t = 0
                for n in range(N):
                    t0 = time.time()
                    _ = _preload_model(model=model, content=test_string, device=device)
                    t1 = time.time()
                    t += (t1 - t0)
                message = f"{(t)/float((N))} for {model} and {device}"
                messages.append(message)
                self.logger.debug(f"{model} {device} vectorise run {N} times.")
                self.logger.info(f"{model} {device} run succesfully!")

        for message in messages:
            self.logger.info(message)
        self.logger.info("completed loading models")


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
        try:
            _ = vectorise(
                model_name=model["model"], 
                model_properties=model["model_properties"], 
                content=content, 
                device=device
            )
        except KeyError as e:
            raise errors.EnvVarError(
                f"Your custom model {model} is missing either `model` or `model_properties`."
                f"""To add a custom model, it must be a dict with keys `model` and `model_properties` as defined in `https://docs.marqo.ai/0.0.20/Advanced-Usage/configuration/#configuring-preloaded-models`"""
            ) from e


class InitializeRedis:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def run(self):
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
        print('\n')


class DownloadFinishText:

    def run(self):
        print('\n')
        print("###########################################################")
        print("###########################################################")
        print("###### !!COMPLETED SUCCESSFULLY!!!         ################")
        print("###########################################################")
        print("###########################################################")
        print('\n')


class MarqoPhrase:

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