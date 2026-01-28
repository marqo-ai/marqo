from typing import List, Optional

from pydantic.v1 import Field, PrivateAttr

from marqo.base_model import StrictBaseModel
from marqo.tensor_search.models.sort_by_model import SortByField


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
    """
    name: str = Field(..., description="The name of the field to collapse on.")
    sort_by: Optional[List[SortByField]] = Field(
        None, alias="sortBy", max_items=1, min_items=1,
        description="List of fields to sort by within the collapse group.",
    )

    _execute: bool = PrivateAttr(False)
    _collapse_filter_string: str = PrivateAttr("")

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
        if not self.should_execute_sort():
            raise RuntimeError(
                "Cannot get collapse filter string when execute sort is disabled"
            )
        return self._collapse_filter_string