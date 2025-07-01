import copy

from marqo.logging import get_logger
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.api_models import SearchQuery, CustomVectorQuery

marqo_query_logger = get_logger('marqo_query')

SECRET_FIELDS = {'imageDownloadHeaders', 'mediaDownloadHeaders', 'modelAuth'}

slow_query_threshold_ms = float(utils.read_env_vars_and_defaults(EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS))
log_query_details = utils.read_env_vars_and_defaults(EnvVars.MARQO_LOG_QUERY_DETAILS).upper() == "TRUE"
log_query_max_length = int(utils.read_env_vars_and_defaults(EnvVars.MARQO_LOG_QUERY_MAX_LENGTH))


class QueryLogger:
    def __init__(self, search_query: SearchQuery):
        self.search_query = search_query
        self.error_logged = False

    @property
    def sanitised_query(self) -> dict:
        """

        """
        query_dict = self.search_query.dict(exclude_none=True, skip_defaults=True, exclude=SECRET_FIELDS)
        q = self.search_query.q

        # Truncate long query strings
        def _truncate_long_query(query_str: str):
            return f'{query_str[:log_query_max_length]}...[truncated:{log_query_max_length}/{len(query_str)}]'

        if isinstance(q, str):
            if len(q) > log_query_max_length:
                query_dict['q'] = _truncate_long_query(q)
        elif isinstance(q, dict):
            has_long_query_string = any([len(key) > log_query_max_length for key in q])
            if has_long_query_string:
                query_dict['q'] = {_truncate_long_query(key) if len(key) > log_query_max_length else key: value
                                   for key, value in q.items()}
        elif isinstance(q, CustomVectorQuery):
            if q.customVector.content and len(q.customVector.content) > log_query_max_length:
                query_dict["q"]["customVector"]["content"] = _truncate_long_query(q.customVector.content)
            # remove custom vector
            query_dict["q"]["customVector"]["vector"] = []

        # remove context vector
        if self.search_query.context and self.search_query.context.tensor:
            sanitised_context_tensor = [{"vector": [], "weight": tensor.weight} for tensor in self.search_query.context.tensor]
            query_dict["context"]["tensor"] = sanitised_context_tensor

        return query_dict

    def log_error_query(self, error_message: str):
        if log_query_details:
            marqo_query_logger.error(f'Failed search query: Error: {error_message}. Query: {self.sanitised_query}')
            self.error_logged = True  # Mark that error was logged

    def log_slow_query(self, elapsed_time_ms: float):
        if log_query_details and not self.error_logged and elapsed_time_ms >= slow_query_threshold_ms:
            marqo_query_logger.warning(f'Slow search query detected: {elapsed_time_ms:.1f}ms. '
                                       f'Query: {self.sanitised_query}')

