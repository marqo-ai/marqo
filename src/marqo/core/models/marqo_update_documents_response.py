from typing import List, Optional

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel


class MarqoUpdateDocumentsItem(MarqoBaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    status: int
    message: Optional[str] = None
    error: Optional[str] = None
    _is_from_vespa_response: Field(type=bool, exclude=True, default=False)

    @root_validator(pre=True)
    def check_status_and_message(cls, values):
        status = values.get('status')
        message = values.get('message')
        error = values.get('error')
        is_from_vespa_response = values.get('_is_from_vespa_response', False)
        # We only want to check the status and message if the response is from Vespa
        if is_from_vespa_response:
            if status == 412:
                message = "Document does not exist in the index."
                status = 404

        # We should put the put error message in both message and error field now as we are not allowed to have
        # break changes. message will be our primary field in the future.
        # TODO remove error in the future
        if error is None:
            error = message
        elif message is None:
            message = error

        values["status"] = status
        values["message"] = message
        values["error"] = error
        return values


class MarqoUpdateDocumentsResponse(MarqoBaseModel):
    errors: bool
    index_name: str
    items: List[MarqoUpdateDocumentsItem]
    processingTimeMs: float
    _success_count: Field(exclude=True, default=0)
    _error_count: Field(exclude=True, default=0)

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
