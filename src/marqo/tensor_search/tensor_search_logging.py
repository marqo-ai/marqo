import logging
from marqo.tensor_search.utils import read_env_vars_and_defaults
from marqo.errors import EnvVarError
def get_logger(name):
    logging.basicConfig()
    logger = logging.getLogger(name)

    log_level = read_env_vars_and_defaults("MARQO_LOG_LEVEL").lower()
    if log_level == "warning":
        logger.setLevel(level=logging.WARNING)
    elif log_level == "info":
        logger.setLevel(level=logging.INFO)
    elif log_level == "debug":
        logger.setLevel(level=logging.DEBUG)
    elif log_level == "error":
        logger.setLevel(level=logging.ERROR)
    else:
        raise EnvVarError(f"The provided {log_level} is not supported."
                          f"The {log_level} should be one of `error`, `warning`, `info`, `debug`."
                          f"Check https://docs.marqo.ai/0.0.13/Advanced-Usage/configuration/ for more info.")

    formatter = logging.Formatter(
        "{asctime} {threadName:>11} {levelname} {message}", style='{')

    return logger