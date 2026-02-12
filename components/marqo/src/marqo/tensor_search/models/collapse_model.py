from typing import List, Optional, ClassVar

from pydantic.v1 import Field, PrivateAttr, validator

from marqo.api.exceptions import InvalidFieldNameError
from marqo.base_model import StrictBaseModel
from marqo.core.unstructured_vespa_index.unstructured_validation import validate_field_name
from marqo.tensor_search.models.sort_by_model import SortOrder


class CollapseSortByField(StrictBaseModel):
    class Config(StrictBaseModel.Config):
        use_enum_values = True
    """
    The model defining the sort by field within the collapse group. No missing policy is needed as this
    sort is for selecting the representative document within each collapse group.

    Attributes:
        field_name (str): The name of the field to sort by.
        order (str): The order of sorting, either 'asc' (ascending) or 'desc' (descending). Defaults to 'desc'.
    """
    field_name: str = Field(..., alias="fieldName", description="The name of the field to sort by.")
    order: SortOrder = SortOrder.Desc

    @validator('field_name')
    def _validate_field_name(cls, v):
        """Validate the field name is in a valid format."""
        try:
            validate_field_name(v)
        except InvalidFieldNameError as e:
            raise ValueError(e)
        return v

class CollapseSortBy(StrictBaseModel):

    """
    The model defining the sort-by configuration within a collapse group. This controls how the
    representative document is selected from each collapse group.

    Attributes:
        fields (List[CollapseSortByField]): List of fields to sort by within the collapse group (max 1).
        num_threads_per_search (Optional[int]): Number of threads to use per search for collapse operation.
        disable_if_main_sort_by_fields (Optional[set[str]]): If the main query is sorted by any of these fields,
            the sortBy feature in the collapse will be disabled to avoid conflicts.
        always_fetch_variants (bool): Whether to always fetch all variants within each collapse group.
            By default (False), only fetch variants if the sort field value is numerical.

    Private Attributes:
        _execute (bool): A flag indicating whether to execute sorting within collapse groups.
        _collapse_filter_string (str): A string representing the collapse filter to be applied in the Vespa query.

    Class Variables:
        COLLAPSE_SORT_BY_QUERY_LIMIT (int): The limit (hits) to be used when generating the Vespa query input
            for collapse sorting. Set to a high number to avoid Vespa's built-in retry mechanism.
    """
    fields: List[CollapseSortByField] = Field(
        ..., min_items=1, max_items=1, description="List of fields to sort by within the collapse group."
    )

    num_threads_per_search: Optional[int] = Field(
        None,
        alias="numThreadsPerSearch",
        description="Number of threads to use per search for collapse operation.",
        ge=1
    )

    disable_if_main_sort_by_fields: Optional[set[str]] = Field(
        None,
        alias="disableIfMainSortByFields",
        description="If the main query is sorted by any of these fields, the sortBy feature in the collapse "
                    "will be disabled to avoid conflicts.",
    )

    always_fetch_variants: bool= Field(
        False, alias="alwaysFetchVariants",
        description=
        "Whether to always fetch all variants within each collapse group. "
        "By default(False), only fetch the sort_by variants if the returned document has "
        "the target collapse sort_by field, and the value of the field is numerical. "
    )

    _execute: bool = PrivateAttr(False)
    _collapse_filter_string: str = PrivateAttr("")

    COLLAPSE_SORT_BY_QUERY_LIMIT: ClassVar[int] = 9999

    def generate_vespa_sort_by_query_input(self):
        return_body = {}
        for field in self.fields:
            return_body[field.field_name] = 1 if field.order == SortOrder.Desc else -1
        return return_body

    def should_execute_sort(self) -> bool:
        return self._execute

    def enable_execute_sort(self):
        self._execute = True

    def disable_execute_sort(self):
        self._execute = False

    def set_collapse_sort_by_filter_string(self, filter_string: str):
        if not self.should_execute_sort():
            raise RuntimeError(
                "Cannot set collapse filter string when execute sort is disabled"
            )
        self._collapse_filter_string = filter_string

    def get_collapse_sort_by_filter_string(self) -> str:
        return self._collapse_filter_string


class CollapseModel(StrictBaseModel):
    """
    The model defining the parameters for collapsing search results based on a specific field. This model
    will be used in the codebase to represent collapse parameters since we only allow one collapse field at the moment.

    Attributes:
        name (str): The name of the field to collapse on.
        sort_by (Optional[CollapseSortBy]): The sort-by configuration for selecting the representative
            document within each collapse group. Contains fields, threading, and variant-fetching options.
    """
    name: str = Field(..., description="The name of the field to collapse on.")
    sort_by: Optional[CollapseSortBy] = Field(
        None, description="List of fields to sort by within the collapse group.",
        alias="sortBy",
    )