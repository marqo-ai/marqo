from typing import Optional, Union, List, Dict, Any

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel
from marqo.core.models.marqo_add_documents_response import BatchResponseStats
from marqo.core.models.marqo_add_documents_response import MarqoBaseDocumentsResponse
from marqo.tensor_search.enums import TensorField


class MarqoGetDocumentsByIdsItem(MarqoBaseModel):
    """A pydantic model for item in MarqoGetDocumentsByIdsResponse.results.

    Only invalid request errors are handled here.
    Valid request should return a dictionary containing the document.
    """
    # This id can be any type as it might be used to hold an invalid id response
    id: Any = Field(alias="_id", default=None)
    status: int
    message: Optional[str] = None
    found: Optional[bool] = Field(alias=str(TensorField.found), default=None)


class MarqoGetDocumentsByIdsResponse(MarqoBaseDocumentsResponse):
    """
    A response from getting documents by their ids from Marqo.
    """
    errors: bool
    results: List[Union[MarqoGetDocumentsByIdsItem, Dict]] = []

    @root_validator(pre=False, skip_on_failure=True)
    def count_items(cls, values):
        results = values.get("results")
        batch_response_count = BatchResponseStats()

        if results:
            for item in results:
                if isinstance(item, dict):
                    # a dictionary indicate a successful response
                    batch_response_count.success_count += 1
                elif isinstance(item, MarqoGetDocumentsByIdsItem):
                    if item.status in range(200, 300):
                        batch_response_count.success_count += 1
                    elif item.status in range(400, 500):
                        batch_response_count.failure_count += 1
                    elif item.status >= 500:
                        batch_response_count.error_count += 1
                    else:
                        raise ValueError(f"Unexpected status code: {item.status}")
                else:
                    raise ValueError(f"Unexpected item type: {type(item)}")

        values['_batch_response_stats'] = batch_response_count
        return values