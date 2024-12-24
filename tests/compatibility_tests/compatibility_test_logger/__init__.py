import logging

def get_logger(name):
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger(name)

    return logger