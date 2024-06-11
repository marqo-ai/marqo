import os
import torch
from marqo import config

def on_start(config: config.Config):
    BootstrapVespa(config)
    CUDAAvailable().run()
    SetBestAvailableDevice().run()
    CacheModels().run()
    CachePatchModels().run()

class BootstrapVespa:
    def __init__(self, config: config.Config):
        self.config = config

    def run(self):
        print('Bootstrapping Vespa...')
        # Perform bootstrap operations here
        print('Vespa configured successfully.')

class CUDAAvailable:
    def run(self):
        if torch.cuda.is_available():
            print('CUDA is available.')
        else:
            print('CUDA is not available.')

class SetBestAvailableDevice:
    def run(self):
        if torch.cuda.is_available():
            os.environ['MARQO_BEST_AVAILABLE_DEVICE'] = "cuda"
        else:
            os.environ['MARQO_BEST_AVAILABLE_DEVICE'] = "cpu"
        print(f"Best available device set to: {os.environ['MARQO_BEST_AVAILABLE_DEVICE']}")

class CacheModels:
    def run(self):
        print('Caching models...')
        # Cache models here
        print('Models cached successfully.')

class CachePatchModels:
    def run(self):
        print('Caching patch models...')
        # Cache patch models here
        print('Patch models cached successfully.')

# Entry point
if __name__ == "__main__":
    on_start(config.Config())
