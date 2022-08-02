import logging


logger = logging.getLogger("marqo.client")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "{asctime} marqo.client {levelname} {message}", style='{')
