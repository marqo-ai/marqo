import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "{asctime} {threadName:>11} {levelname} {message}", style='{')

    return logger