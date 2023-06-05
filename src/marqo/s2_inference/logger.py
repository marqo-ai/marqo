import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()  # Creates a new handler
    formatter = logging.Formatter("{asctime} {threadName:>11} {levelname} {message}", style='{')
    handler.setFormatter(formatter)  # Applies the formatter to the handler

    logger.addHandler(handler)  # Adds the handler to the logger

    return logger
