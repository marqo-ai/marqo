from typing import Optional, Union, List, Dict

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel
from marqo.tensor_search.enums import TensorField


class MarqoGetDocumentsByIdsItem(MarqoBaseModel):
    """A pydantic model for item in MarqoGetDocumentsByIdsResponse.results.

    Only invalid request errors are handled here.
    Valid request should return a dictionary containing the document.
    """
    id: Optional[str] = Field(alias="_id", default=None)
    status: int
    message: Optional[str] = None
    found: Optional[bool] = Field(alias=str(TensorField.found), default=None)


class MarqoGetDocumentsByIdsResponse(MarqoBaseModel):
    """
    A response from getting documents by their ids from Marqo.
    """
    errors: bool
    results: List[Union[MarqoGetDocumentsByIdsItem, Dict]] = []
    success_count: int = Field(exclude=True, default=0)
    error_count: int = Field(exclude=True, default=0)
    failure_count: int = Field(exclude=True, default=0)

    @root_validator(pre=False, skip_on_failure=True)
    def count_items(cls, values):
        results = values.get("results")
        if results:
            for item in results:
                if isinstance(item, dict):
                    # a dictionary indicate a successful response
                    values["success_count"] += 1
                elif isinstance(item, MarqoGetDocumentsByIdsItem):
                    if item.status in range(200, 300):
                        values["success_count"] += 1
                    elif item.status in range(400, 500):
                        values["failure_count"] += 1
                    elif item.status >= 500:
                        values["error_count"] += 1
                    else:
                        raise ValueError(f"Unexpected status code: {item.status}")
                else:
                    raise ValueError(f"Unexpected item type: {type(item)}")
        return values
