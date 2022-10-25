from marqo.tensor_search.tensor_search_logging import get_logger
import time
from marqo.tensor_search import index_meta_cache
from marqo import config
from marqo.tensor_search.web import api_utils
from marqo._httprequests import HttpRequests
from marqo import errors


def on_start(marqo_os_url: str):
        
    to_run_on_start = (
                        PopulateCache(marqo_os_url),
                        DownloadStartText(),
                        CUDAAvailable(), 
                        ModelsForCacheing(), 
                        NLTK(), 
                        DownloadFinishText(),
                        MarqoWelcome(),
                        MarqoPhrase()
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
                        "        - If you are using an external Marqo-os instance, check if it is running: `curl <YOUR MARQO-OS URL>` \n"
                        "        - Ensure that the OPENSEARCH_URL environment variable defined in the `docker run marqo` command points to Marqo-os\n",
                link="https://github.com/marqo-ai/marqo/tree/mainline/src/marqo#c-build-and-run-the-marqo-as-a-docker-container-connecting-to-marqo-os-which-is-running-on-the-host"
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

    def __init__(self):
        
        pass

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
            device_names.append( {'id':device_id, 'name':id_to_device(device_id)})
        self.logger.info(f"found devices {device_names}")


class NLTK: 

    """predownloads the nltk stuff
    """

    logger = get_logger('NLTK setup')

    def __init__(self):

        pass 

    def run(self):
        self.logger.info("downloading nltk data....")
        import nltk

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')        
        self.logger.info("completed loading nltk")


class ModelsForCacheing:
    
    """warms the in-memory model cache by preloading good defaults
    """
    logger = get_logger('ModelsForStartup')

    def __init__(self):
        import torch
      
        self.models = (
            'onnx/all_datasets_v4_MiniLM-L6',
            "ViT-L/14",
        )
        # TBD to include cross-encoder/ms-marco-TinyBERT-L-2-v2

        self.default_devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']

        self.logger.info(f"pre-loading {self.models} onto devices={self.default_devices}")

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
                message = f"{(t)/float((N))} for {model} and {device}"
                messages.append(message)
                self.logger.info(f"{model} {device} run succesfully!")

        for message in messages:
            self.logger.info(message)
        self.logger.info("completed loading models")


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
        print("###### !!COMPLETED SUCCESFULLY!!!          ################")
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
