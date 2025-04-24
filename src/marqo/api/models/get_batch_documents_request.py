from pydantic.v1 import Field, conlist

from marqo.base_model import MarqoBaseModel


class GetBatchDocumentsRequest(MarqoBaseModel):
    """
    A request model for getting batch documents by their ids from Marqo.
    """
    document_ids: conlist(str, min_items=1) = Field(alias="documentIds")