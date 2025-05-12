import logging.config
import os
from enum import Enum

from marqo import marqo_docs
from marqo.api.configs import default_env_vars
from marqo.api.exceptions import EnvVarError
from marqo.tensor_search.enums import EnvVars


class LogLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class LogFormat(str, Enum):
    PLAIN = "plain"
    JSON = "json"


# Please note that calling os.environ directly is required to avoid cyclic dependency
raw_log_level = os.environ.get(EnvVars.MARQO_LOG_LEVEL, default_env_vars()[EnvVars.MARQO_LOG_LEVEL])
raw_log_format = os.environ.get(EnvVars.MARQO_LOG_FORMAT, default_env_vars()[EnvVars.MARQO_LOG_FORMAT])

try:
    LOG_LEVEL = LogLevel(raw_log_level.lower()).name  # need uppercase level name in the config
except ValueError:
    raise EnvVarError(f"The provided environment variable `{EnvVars.MARQO_LOG_LEVEL}` = `{raw_log_level}` is not "
                      f"supported. The environment variable `{EnvVars.MARQO_LOG_LEVEL}` should be one of "
                      f"{', '.join([l for l in LogLevel])}. Check {marqo_docs.configuring_marqo()} for more info.")

try:
    LOG_FORMAT = LogFormat(raw_log_format.lower()).value
except ValueError:
    raise EnvVarError(f"The provided environment variable `{EnvVars.MARQO_LOG_FORMAT}` = `{raw_log_format}` is not "
                      f"supported. The environment variable `{EnvVars.MARQO_LOG_FORMAT}` should be one of "
                      f"{', '.join([f for f in LogFormat])}. Check {marqo_docs.configuring_marqo()} for more info.")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Allows integration with third-party loggers
    "formatters": {
        "default-plain": {
            "format": "[%(asctime)s] %(process)d %(levelname)s %(name)s: %(message)s"
        },
        "default-json": {
            "()": "pythonjsonlogger.orjson.OrjsonFormatter",
            "fmt": "%(asctime) %(process) %(levelname) %(name) %(message)",
            "rename_fields": {
                "asctime": "timestamp",
                "levelname": "level",
            },
            "exc_info_as_array": True,
            "stack_info_as_array": True,
        },
        "access-plain": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '[%(asctime)s] %(process)d %(levelname)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
        "access-json": {
            "()": "uvicorn.logging.AccessFormatter",
            "format": (
                '{"timestamp": "%(asctime)s", '
                '"process": "%(process)d", '
                '"level": "%(levelname)s", '
                '"client_addr": "%(client_addr)s", '
                '"request_line": "%(request_line)s", '
                '"status_code": "%(status_code)s"}'
            )
        }
    },
    "handlers": {
        "default": {
            "formatter": f"default-{LOG_FORMAT}",
            "class": "logging.StreamHandler",
        },
        "access": {
            "formatter": f"access-{LOG_FORMAT}",
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
        },
        "httpcore": {
            "handlers": ["default"],
            "level": LOG_LEVEL if LOG_LEVEL == "ERROR" else "WARNING",  # mute verbose httpcore info level log
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
