from typing import List, Optional, ClassVar

from pydantic.v1 import Field, PrivateAttr, root_validator

from marqo.base_model import StrictBaseModel
from marqo.tensor_search.models.sort_by_model import SortOrder


class CollapseSortByField(StrictBaseModel):
    """
    The model defining the sort by field within the collapse group. No missing policy is needed as this
    sort is for selecting the representative document within each collapse group.

    Attributes:
        field_name (str): The name of the field to sort by.
        order (str): The order of sorting, either 'asc' (ascending) or 'desc' (descending). Defaults to 'desc'.
    """
    field_name: str = Field(..., alias="fieldName", description="The name of the field to sort by.")
    order: SortOrder = SortOrder.Desc


class CollapseModel(StrictBaseModel):
    """
    The model defining the parameters for collapsing search results based on a specific field. This model
    will be used in the codebase to represent collapse parameters since we only allow one collapse field at the moment.

    Attributes:
        name (str): The name of the field to collapse on.
        sort_by (Optional[List[SortByField]]): List of fields to sort by within the collapse group. The highest
        ranked document in each collapse group will be returned as the representative document for that group.

    Private Attributes:
        _execute (bool): A flag indicating whether to execute sorting within collapse groups. It will be used when
        generating the Vespa query input.
        _collapse_filter_string (str): A string representing the collapse filter to be applied in the Vespa query.

    Class Variables:
        COLLAPSE_SORT_BY_QUERY_LIMIT (int): The limit(hits) to be used when generating the Vespa query input for collapse sorting.
        We deliberately set it to a high number to avoid the built-in retry mechanism of the Vespa collapse feature.
        By default, Vespa wll retry 4 times with increasing hits (default hits: 10, then 50, 250, 1250, 6250).
        However, it will break if totalHits is smaller than the hits used in the query.
        Therefore, we set a high limit to have an early termination of the retry mechanism.
    """
    name: str = Field(..., description="The name of the field to collapse on.")
    sort_by: Optional[List[CollapseSortByField]] = Field(
        None, alias="sortBy", max_items=1, min_items=1,
        description="List of fields to sort by within the collapse group.",
    )
    num_threads_per_search: Optional[int] = Field(
        1,
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

    _execute: bool = PrivateAttr(False)
    _collapse_filter_string: str = PrivateAttr("")

    COLLAPSE_SORT_BY_QUERY_LIMIT: ClassVar[int] = 9999

    def generate_vespa_sort_by_query_input(self):
        if self.sort_by is None:
            return None
        return_body = {}
        for field in self.sort_by:
            return_body[field.field_name] = 1 if field.order == "desc" else -1
        return return_body

    def should_execute_sort(self) -> bool:
        return self._execute

    def enable_execute_sort(self):
        self._execute = True

    def disable_execute_sort(self):
        self._execute = False

    def set_collapse_filter_string(self, filter_string: str):
        if not self.should_execute_sort():
            raise RuntimeError(
                "Cannot set collapse filter string when execute sort is disabled"
            )
        self._collapse_filter_string = filter_string

    def get_collapse_filter_string(self) -> str:
        return self._collapse_filter_string

    @root_validator(pre=False)
    def validate_num_threads_per_search(cls, values):
        num_threads_per_search = values.get("num_threads_per_search")
        sort_by = values.get("sort_by")

        if num_threads_per_search is not None and sort_by is None:
            raise ValueError(
                "numThreadsPerSearch is set but sortBy is not provided. "
                "numThreadsPerSearch can only be set when sortBy(collapseField) is provided "
            )