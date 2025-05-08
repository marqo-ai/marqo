import logging

def get_logger(name):
    # Configure root logger only once
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(format='[%(asctime)s] %(levelname)s %(name)s:%(lineno)d %(message)s', level=logging.INFO)
    
    # Get the logger with the provided name
    logger = logging.getLogger(name)
    
    # Set DEBUG level for loggers in the tests.compatibility_tests package
    if name.startswith('tests.compatibility_tests'):
        logger.setLevel(logging.DEBUG)
    
    return logger