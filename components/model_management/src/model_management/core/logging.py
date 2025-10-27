import logging
import logging.config

from model_management.core.settings import Settings


def instantiate_logger(s: Settings) -> None:
    LOG_FORMAT = s.marqo_log_format.value  # e.g. "PLAIN" in upper case
    LOG_LEVEL = s.marqo_log_level.value  # e.g. "INFO" in upper case

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default-plain": {
                "format": "[%(asctime)s] %(process)d %(levelname)s %(name)s: %(message)s"
            },
            "default-json": {
                "()": "pythonjsonlogger.orjson.OrjsonFormatter",
                "fmt": "%(asctime) %(process) %(levelname) %(name) %(message)",
                "rename_fields": {"asctime": "timestamp", "levelname": "level"},
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
                ),
            },
        },
        "handlers": {
            "default": {
                "formatter": f"default-{LOG_FORMAT.lower()}",
                "class": "logging.StreamHandler",
            },
            "access": {
                "formatter": f"access-{LOG_FORMAT.lower()}",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": LOG_LEVEL,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
            "httpx": {
                "handlers": ["default"],
                "level": LOG_LEVEL if LOG_LEVEL == "ERROR" else "WARNING",
                "propagate": False,
            },
            "httpcore": {
                "handlers": ["default"],
                "level": LOG_LEVEL if LOG_LEVEL == "ERROR" else "WARNING",
                "propagate": False,
            },
            "marqo_query": {
                "handlers": ["default"],
                "level": "WARNING",
                "propagate": False,
            },
        },
        "root": {"handlers": ["default"], "level": LOG_LEVEL},
    }
    logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger instance by name.

    Args:
        name (str): The name of the logger to retrieve.

    Returns:
        logging.Logger: The logger instance associated with the given name.
    """
    return logging.getLogger(name)
