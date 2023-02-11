import logging
from marqo.tensor_search.utils import read_env_vars_and_defaults
def get_logger(name):


    log_level = read_env_vars_and_defaults("LOG_LEVEL")
    if log_level == "warning":
        logging.basicConfig(level=logging.WARNING)
    elif log_level == "info":
        logging.basicConfig(level=logging.INFO)


    #logging.basicConfig()
    logger = logging.getLogger(name)
    #logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "{asctime} {threadName:>11} {levelname} {message}", style='{')

    return logger