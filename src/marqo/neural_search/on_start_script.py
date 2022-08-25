# use this function to pre-load when marqo service starts
from marqo.neural_search.neural_search_logging import get_logger

def on_start():

    # pre load the modesl into the model cache
    models_loaded_into_cache = ModelsForCacheing()
    models_loaded_into_cache.run()

class NLTK:

    logger = get_logger('NLTK setup')

    def __init__(self):

        pass 

    def run(self):
        import nltk

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')        
        self.logger.info("completed loading nltk")

class ModelsForCacheing:
    
    

    logger = get_logger('ModelsForStartup')

    def __init__(self):

        self.models = (
            'hf/all_datasets_v4_MiniLM-L6',
            'onnx/all_datasets_v4_MiniLM-L6',
            "ViT-L/14",
        )
        self.default_device = 'cpu'

        self.logger.info(f"pre-loading {self.models} onto device={self.default_device}")

    def run(self):
        from marqo.s2_inference.s2_inference import vectorise
        for model in self.models:
            test_string = 'this is a test string'
            _ = vectorise(model, test_string, device=self.default_device)
        self.logger.info("completed loading models")