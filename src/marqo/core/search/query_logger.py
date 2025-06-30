from marqo.logging import get_logger
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars

marqo_query_logger = get_logger('marqo_query')

slow_query_threshold_ms = float(utils.read_env_vars_and_defaults(EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS))
log_query_details = utils.read_env_vars_and_defaults(EnvVars.MARQO_LOG_QUERY_DETAILS).upper() == "TRUE"


class QueryLogger:
    def __init__(self, query_dict):
        self.query_dict = query_dict
        self.error_logged = False

    @property
    def sanitised_query(self):
        return self.query_dict # TODO sanitise it

    def log_error_query(self, error_message: str):
        if log_query_details:
            marqo_query_logger.error(f'Failed search query: Error: {error_message}. Query: {self.sanitised_query}')
            self.error_logged = True  # Mark that error was logged

    def log_slow_query(self, elapsed_time_ms: float):
        if log_query_details and not self.error_logged and elapsed_time_ms >= slow_query_threshold_ms:
            marqo_query_logger.warning(f'Slow search query detected: {elapsed_time_ms:.1f}ms. '
                                       f'Query: {self.sanitised_query}')

