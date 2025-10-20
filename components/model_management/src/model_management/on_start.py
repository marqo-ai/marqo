from .config import Config
from .core.settings import Settings


def on_start(cfg: Config, settings: Settings):
    to_run_on_start = (PreLoadModels(cfg, settings), WelcomeMessage())

    for thing_to_start in to_run_on_start:
        thing_to_start.run()


class PreLoadModels(object):
    def __init__(self, cfg: Config, settings: Settings):
        self.config = cfg
        self.settings = settings

    def run(self):
        if self.settings.marqo_models_to_preload:
            for model in self.settings.marqo_models_to_preload:
                self.config.model_manager.load_model(model)


class WelcomeMessage:
    def run(self):
        message = r"""
        +----------------------------------------------------------------------------------------------+
        |                                                                                              |
        |                           Welcome to Marqo Model Management Container                        |
        |                                                                                              |
        |                                                                                              |
        +----------------------------------------------------------------------------------------------+
        """
        print(message, flush=True)
