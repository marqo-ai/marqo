import logging
from marqo.tensor_search.utils import read_env_vars_and_defaults
def get_logger(name):
    logging.basicConfig()
    logger = logging.getLogger(name)

    log_level = read_env_vars_and_defaults("MARQO_LOG_LEVEL")
    if log_level == "warning":
        logging.basicConfig(level=logging.WARNING)
    elif log_level == "info":
        logging.basicConfig(level=logging.INFO)

    formatter = logging.Formatter(
        "{asctime} {threadName:>11} {levelname} {message}", style='{')

    return logger