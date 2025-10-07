from enum import StrEnum


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(StrEnum):
    PLAIN = "PLAIN"
    JSON = "JSON"
