from enum import Enum
from pydantic.v1 import Field, validator
from typing import List, Optional

from marqo.api.exceptions import InvalidFieldNameError
from marqo.base_model import StrictBaseModel
from marqo.core.unstructured_vespa_index.unstructured_validation import validate_field_name


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
    class Config(StrictBaseModel.Config):
        use_enum_values = True

    field_name: str = Field(alias="fieldName")
    order: SortOrder = SortOrder.Desc
    missing: SortMissingPolicy = SortMissingPolicy.Last

    @validator('field_name')
    def _validate_field_name(cls, v):
        """Validate the field name is in a valid format."""
        try:
            validate_field_name(v)
        except InvalidFieldNameError as e:
            raise ValueError(e)
        return v


class SortByModel(StrictBaseModel):
    """
    The SortByModel defines how to sort the results of a search query.

    Attributes:
        fields (List[SortByField]): A list of SortByField objects that define the fields to sort by.
            Note that the order of fields in this list determines the order of sorting. Fields presented later will
            be used as tiebreakers for fields presented earlier.
        sort_depth (Optional[int]): The depth of sorting at the global phase.
            Check Vespa Custom Searcher for more details.
        min_sort_candidates (Optional[int]): The minimum number of candidates to be retrieved.
            Check Vespa Custom Searcher for more details.
    """
    fields: List[SortByField] = Field(..., min_items=1, max_items=3)
    sort_depth: Optional[int] = Field(None, ge=1, alias="sortDepth")
    min_sort_candidates: Optional[int] = Field(None, ge=1, alias="minSortCandidates")