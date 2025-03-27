import logging.config
import os

from marqo import marqo_docs
from marqo.api.configs import default_env_vars
from marqo.api.exceptions import EnvVarError
from marqo.tensor_search.enums import EnvVars


# TODO confirm why we don't accept CRITICAL as a valid option
VALID_LOG_LEVELS = {"ERROR", "WARNING", "INFO", "DEBUG"}

# Please note that calling os.environ directly is required to avoid cyclic dependency
raw_log_level = os.environ.get(EnvVars.MARQO_LOG_LEVEL, default_env_vars()[EnvVars.MARQO_LOG_LEVEL])

if raw_log_level.upper() not in VALID_LOG_LEVELS:
    raise EnvVarError(f"The provided environment variable `{EnvVars.MARQO_LOG_LEVEL}` = `{raw_log_level}` is not "
                      f"supported. The environment variable `{EnvVars.MARQO_LOG_LEVEL}` should be one of `error`, "
                      f"`warning`, `info`, `debug`. Check {marqo_docs.configuring_marqo()} for more info.")

LOG_LEVEL = raw_log_level.upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Allows integration with third-party loggers
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(process)d %(levelname)s %(name)s: %(message)s"
        },
        # TODO json format is a better option for some log collection tool, define it here for future use
        "json": {
            "format": (
                '{"timestamp": "%(asctime)s", '
                '"process": "%(process)d", '
                '"level": "%(levelname)s", '
                '"name": "%(name)s", '
                '"message": "%(message)s"}'
            )
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '[%(asctime)s] %(process)d %(levelname)s %(client_addr)s - "%(request_line)s" %(status_code)s',  # noqa: E501
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",  # Change to "json" for structured logging output if needed
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False
        },
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",  # access log should be printed out even when root log level is higher than info
            "propagate": False,
        },
        "httpx": {
            "handlers": ["default"],
            "level": LOG_LEVEL if LOG_LEVEL == "ERROR" else "WARNING",  # mute verbose httpx info level log
            "propagate": False,
        }
    },
    "root": {
        "handlers": ["default"],
        "level": LOG_LEVEL,
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

# Define an alias of getLogger, so we minimise the change in Marqo code
get_logger = logging.getLogger
