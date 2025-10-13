from marqo import config, version
from marqo import marqo_docs
from marqo.logging import get_logger
from marqo.tensor_search import index_meta_cache, utils
from marqo.tensor_search.enums import EnvVars

logger = get_logger(__name__)


def on_start(config: config.Config):
    to_run_on_start = (
        BootstrapVespa(config),
        PopulateCache(config),
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


class PopulateCache:
    """Populates the cache on start"""

    def __init__(self, config: config.Config):
        self.config = config

    def run(self):
        logger.debug('Starting index cache refresh thread')
        index_meta_cache.start_refresh_thread(self.config)


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
