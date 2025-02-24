"""
This file contains the `VespaErrorHandlingMixin` class, which provides error-handling methods
for dealing with **unexpected HTTP responses** from Vespa.

Functions Included:
--------------------
1. **Error Handling**
   - `_raise_for_status()` → Raises an error for HTTP responses that are not 2xx.
   - `_raise_for_error_code()` → Maps Vespa error codes to custom exceptions.
   - `_query_raise_for_status()` → Handles query-related HTTP errors.
   - `_is_timeout_error()` → Determines if an error is due to a timeout.
   - `translate_vespa_document_response()` → Translates Vespa document responses into Marqo document responses.

This mixin ensures **graceful error handling** across different operations.

Usage Example:
--------------
try:
    vespa.query()
except VespaStatusError as e:
    print(f"Query failed: {e}")
"""
from typing import TYPE_CHECKING, Optional, Tuple

import httpx

from marqo.vespa.exceptions import (
    VespaStatusError, VespaError, VespaTimeoutError, InvalidVespaApplicationError, VespaActivationConflictError
)
from marqo.vespa.models import QueryResult, Error
import marqo.logging
if TYPE_CHECKING:
    from ._client_base import VespaClientBase

logger = marqo.logging.get_logger(__name__)

class VespaErrorHandlingMixin:
    _VESPA_ERROR_CODE_TO_EXCEPTION = {
        'INVALID_APPLICATION_PACKAGE': InvalidVespaApplicationError,
        'ACTIVATION_CONFLICT': VespaActivationConflictError
    }

    def _raise_for_status(self: "VespaClientBase", resp: httpx.Response) -> None:
        """Take the response and raise an VespaStatusError if the status code is not 2xx.

        Args:
            resp: The response object from the httpx client

        Raises:
            VespaStatusError: If the status code is not 2xx
        """
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            response = e.response
            try:
                json = response.json()
                error_code = json['error-code']
                message = json['message']
            except Exception:
                raise VespaStatusError(message=response.text, cause=e) from e

            self._raise_for_error_code(error_code, message, e)

    def _raise_for_error_code(self: "VespaClientBase", error_code: str, message: str, cause: Exception) -> None:
        exception = self._VESPA_ERROR_CODE_TO_EXCEPTION.get(error_code, VespaError)
        if exception:
            raise exception(message=message, cause=cause) from cause

        raise VespaStatusError(message=f'{error_code}: {message}', cause=cause) from cause

    def _query_raise_for_status(self: "VespaClientBase", resp: httpx.Response) -> None:
        """
        Query API specific raise for status method.
        If multiple errors:
            If all errors are timeout, raise VespaTimeoutError (504).
            If even one error is not timeout, raise VespaStatusError (500).
        """
        # See error codes here https://github.com/vespa-engine/vespa/blob/master/container-core/src/main/java/com/yahoo/container/protect/Error.java
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                result = QueryResult(**resp.json())
                if (
                        result.root.errors is not None
                        and len(result.root.errors) > 0
                ):
                    for error in result.root.errors:
                        if not self._is_timeout_error(error, resp):
                            # Raise 500 if any error is not timeout
                            raise VespaStatusError(message=resp.text, cause=e) from e
                    # Raise 504 if all errors are timeout
                    raise VespaTimeoutError(message=resp.text, cause=e) from e
                raise e
            except VespaStatusError:
                raise
            except Exception:
                raise VespaStatusError(message=resp.text, cause=e) from e

    @classmethod
    def _is_timeout_error(cls, error: Error, resp: httpx.Response) -> bool:
        """
        Check if the query error is a timeout error.
        """

        if error.code == 8 and error.message == "Search request soft doomed during query setup and initialization.":
            logger.warn('Detected soft doomed query')
            return True
        if error.code == 12 and resp.status_code == 504:
            return True

        return False

    def translate_vespa_document_response(self, status: int, message: Optional[str] = None) -> Tuple[
        int, Optional[str]]:
        """A helper function to translate Vespa document response into the expected status, message that
        is used in Marqo document API responses.

        Args:
            status: The status code from Vespa document response
            message: The message from Vespa document response

        Return:
            A tuple of status code and the message in the response
        """
        vespa_status_code_to_marqo_doc_error_map = {
            200: (200, None),
            404: (404, "Document does not exist in the index"),
            # Update documents get 412 from Vespa for document not found as we use condition
            412: (404, "Document does not exist in the index"),
            429: (429, "Marqo vector store receives too many requests. Please try again later"),
            507: (400, "Marqo vector store is out of memory or disk space"),
        }

        if status in vespa_status_code_to_marqo_doc_error_map:
            return vespa_status_code_to_marqo_doc_error_map[status]
        elif status == 400 and isinstance(message, str) and "could not parse field" in message.lower():
            # TODO Block the invalid special characters before sending to Vespa
            return 400, f"The document contains invalid characters in the fields. Original error: {message} "
        else:
            logger.error(
                f"An unexpected error occurred from the Vespa document response. "
                f"status: {status}, message: {message}"
                )
            return 500, f"Marqo vector store returns an unexpected error with this document. Original error: {message}"