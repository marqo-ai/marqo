import copy

from marqo.logging import get_logger
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.api_models import SearchQuery, CustomVectorQuery

marqo_query_logger = get_logger('marqo_query')

SECRET_FIELDS = ['imageDownloadHeaders', 'image_download_headers', 'mediaDownloadHeaders', 'modelAuth']

slow_query_threshold_ms = float(utils.read_env_vars_and_defaults(EnvVars.MARQO_SLOW_QUERY_THRESHOLD_MS))
log_query_details = utils.read_env_vars_and_defaults(EnvVars.MARQO_LOG_QUERY_DETAILS).upper() == "TRUE"
log_query_max_length = int(utils.read_env_vars_and_defaults(EnvVars.MARQO_LOG_QUERY_MAX_LENGTH))


class QueryLogger:
    def __init__(self, search_query: SearchQuery):
        self.search_query = search_query
        self.error_logged = False

    @property
    def sanitised_query(self) -> dict:
        query_dict = self.search_query.dict(exclude_none=True, skip_defaults=True)

        # query = self.query_dict.get('q', None)
        #
        # # Replace secret fields with empty dict
        # for field in SECRET_FIELDS:
        #     if field in self.query_dict:
        #         query_updates[field] = {}
        #
        # # Truncate long query strings
        # def _truncate_long_query(query_str: str):
        #     return f'[truncated:{log_query_max_length}/{len(query_str)}] {query_str[:log_query_max_length]}'
        #
        # if isinstance(query, str):
        #     if len(query) > log_query_max_length:
        #         query_updates['q'] = _truncate_long_query(query)
        # elif isinstance(query, dict):
        #     if 'customVector' in query:
        #         pass
        #     else:
        #         has_long_query_string = any([len(key) > log_query_max_length for key in query])
        #         if has_long_query_string:
        #             query_updates['q'] = {_truncate_long_query(key) if len(key) > log_query_max_length else key: value
        #                                   for key, value in query.items()}

        # sanitise customer vector
        if isinstance(self.search_query.q, CustomVectorQuery):
            query_dict["q"]["customVector"]["vector"] = []

        # sanitise context tensor
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

