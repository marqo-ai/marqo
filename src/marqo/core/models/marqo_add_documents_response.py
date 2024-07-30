from typing import List, Optional

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel


class MarqoAddDocumentsItem(MarqoBaseModel):
    """A response from adding a document to Marqo.

    This model takes the response from Marqo vector store and translate it to a user-friendly response.
    """
    status: int
    id: Optional[str] = Field(alias="_id", default=None)
    message: Optional[str] = None
    error: Optional[str] = None
    code: Optional[str] = None
    _is_from_vespa_response: Field(type=bool, exclude=True, default=False)

    @root_validator(pre=True)
    def check_status_and_message(cls, values):
        status = values.get('status')
        message = values.get('message')
        error = values.get('error')
        is_from_vespa_response = values.get('_is_from_vespa_response', False)
        # We only want to check the status and message if the response is from Vespa
        if is_from_vespa_response:
            if not isinstance(status, int):
                raise ValueError(f"status must be an integer, got {status}")
            if status == 200:
                pass
            elif status == 429:
                message = "Marqo vector store receives too many requests. Please try again later."
            elif status == 507:
                message = "Marqo vector store is out of memory or disk space."
            else:
                status = 500

        # We should put the put error message in both message and error field now as we are not allowed to have
        # break changes. message will be our primary field in the future.
        if error is None:
            error = message
        elif message is None:
            message = error

        values["status"] = status
        values["message"] = message
        values["error"] = error
        return values


class MarqoAddDocumentsResponse(MarqoBaseModel):
    errors: bool
    processingTimeMs: float
    index_name: str  # TODO Change this to camelCase in the future (Breaking change!)
    items: List[MarqoAddDocumentsItem]
    _success_count: Field(type=int, exclude=True, default=0)
    _error_count: Field(type=int, exclude=True, default=0)

    @root_validator(skip_on_failure=True)
    def calculate_success_and_error_count(cls, values):
        items = values.get('items')
        success_count = 0
        error_count = 0
        if items:
            for item in items:
                if 200 <= item.status < 500 or items.status == 507:
                    success_count += 1
                elif 400 <= item.status < 600:
                    error_count += 1
                else:
                    raise ValueError(f"Invalid status code: {item.status}")
        values["_success_count"] = success_count
        values["_error_count"] = error_count
        return values

    def get_success_count(self) -> int:
        return self._success_count

    def get_error_count(self) -> int:
        return self._error_count
