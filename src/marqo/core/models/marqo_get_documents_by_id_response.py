from typing import Optional, Union, List, Dict

from pydantic import Field

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
    found: Optional[bool] = Field(aliase=str(TensorField.found), default=None)


class MarqoGetDocumentsByIdsResponse(MarqoBaseModel):
    """
    A response from getting documents by their ids from Marqo.
    """
    errors: bool
    results: List[Union[MarqoGetDocumentsByIdsItem, Dict]] = []
