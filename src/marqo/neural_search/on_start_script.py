# use this function to pre-load when marqo service starts
from marqo.neural_search.neural_search_logging import get_logger

def on_start():

    to_run_on_start = (CUDAAvailable(), ModelsForCacheing(), NLTK())

    for thing_to_start in to_run_on_start:
        thing_to_start.run()


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
            device_names += id_to_device(device_id)
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

        self.models = (
            'hf/all_datasets_v4_MiniLM-L6',
            'onnx/all_datasets_v4_MiniLM-L6',
            "ViT-B/16",
        )
        # TBD to include cross-encoder/ms-marco-TinyBERT-L-2-v2

        self.default_devices = ['cpu']

        self.logger.info(f"pre-loading {self.models} onto devices={self.default_devices}")

    def run(self):
        from marqo.s2_inference.s2_inference import vectorise
        test_string = 'this is a test string'

        for model in self.models:
            for device in self.default_devices:
                _ = vectorise(model, test_string, device=device)
        self.logger.info("completed loading models")