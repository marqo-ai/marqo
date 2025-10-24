from typing import Union

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
    """
    This class logs out sanitised the full search query of slow or failed search requests.
    It logs the query to a dedicated logger so we can redirect the log later
    """
    def __init__(self, search_query: SearchQuery):
        self.search_query = search_query
        self.error_logged = False

    @property
    def sanitised_query(self) -> dict:
        """
        This method sanitises the query object by
        * Generating a dictionary from the search query object to avoid changing the original query
        * Excluding None values and skip default values
        * Excluding fields containing secrets like download headers and model auth
        * Removing vectors from custom vector search and tensor context
        * Truncating long query string
        """
        query_dict = self.search_query.dict(by_alias=True, exclude_none=True, skip_defaults=True, exclude=SECRET_FIELDS)
        q = self.search_query.q

        # Truncate long query strings
        def _truncate_if_long(query_str: str) -> str:
            if len(query_str) > log_query_max_length:
                return f'{query_str[:log_query_max_length]}...[truncated:{log_query_max_length}/{len(query_str)}]'
            else:
                return query_str

        def _sanitise_str_or_dict_query(query: Union[str, dict]) -> Union[str, dict]:
            if isinstance(query, str):
                return _truncate_if_long(query)
            elif isinstance(query, dict):
                return {_truncate_if_long(key): value for key, value in query.items()}
            else:
                return query

        if isinstance(q, CustomVectorQuery):
            if q.customVector.content:
                query_dict["q"]["customVector"]["content"] = _truncate_if_long(q.customVector.content)
            # remove custom vector
            query_dict["q"]["customVector"]["vector"] = []
        elif isinstance(q, (str, dict)):
            query_dict['q'] = _sanitise_str_or_dict_query(q)
        else: # q is None, handle separate tensor and lexical q in hybrid parameter
            if self.search_query.hybridParameters.queryTensor:
                query_dict["hybridParameters"]["queryTensor"] = _sanitise_str_or_dict_query(
                    self.search_query.hybridParameters.queryTensor)
            if self.search_query.hybridParameters.queryLexical:
                query_dict["hybridParameters"]["queryLexical"] = _sanitise_str_or_dict_query(
                    self.search_query.hybridParameters.queryLexical)

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
        if log_query_details and not self.error_logged and elapsed_time_ms > slow_query_threshold_ms:
            marqo_query_logger.warning(f'Slow search query detected: {elapsed_time_ms:.1f}ms. '
                                       f'Query: {self.sanitised_query}')
