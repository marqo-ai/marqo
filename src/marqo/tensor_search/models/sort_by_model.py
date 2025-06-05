from enum import Enum
from pydantic.v1 import Field
from typing import List, Optional

from marqo.base_model import StrictBaseModel


class SortOrder(str, Enum):
    Asc = "asc"
    Desc= "desc"


class SortMissingPolicy(str, Enum):
    Last = "last"
    First = "first"



class SortByField(StrictBaseModel):
    """
    The sort by field model defines how to sort the results based on the target field.

    Attributes:
        field_name (str): The name of the field to sort by.
        order (SortOrder): The order of sorting, either asc(ascending) or desc(descending). Defaults to desc.
        missing (SortMissingPolicy): Defines how to handle missing values in the sort field. Defaults to last.
    """
    field_name: str = Field(alias="fieldName")
    order: SortOrder = SortOrder.Desc
    missing: SortMissingPolicy = SortMissingPolicy.Last


class SortByModel(StrictBaseModel):
    """
    The SortByModel defines how to sort the results of a search query.

    Attributes:
        fields (List[SortByField]): A list of SortByField objects that define the fields to sort by.
            Note that the order of fields in this list determines the order of sorting. Fields presented later will
            be used as tiebreakers for fields presented earlier.
        sortDepth (Optional[int]): The depth of sorting at the global phase.
            Check Vespa Customer Searcher for more details.
        minSortCandidates (Optional[int]): The minimum number of candidates to be retrieved.
            Check Vespa Customer Searcher for more details.
    """
    fields: List[SortByField]
    sortDepth: Optional[int] = Field(None, ge=1)
    minSortCandidates: Optional[int] = Field(None, ge=1)