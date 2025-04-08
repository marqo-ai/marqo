from typing import List, Optional, Literal, Dict
from marqo.base_model import StrictBaseModel
from pydantic import Field, validator

class RangeConfiguration(StrictBaseModel):
    class Config:
        allow_population_by_field_name = False # disable ability to pass from_ or to_, only alias

    from_: Optional[float] = Field(None, alias="from")
    to_: Optional[float] = Field(None, alias="to")
    name: Optional[str] = None

    @validator('to_')
    def validate_range(cls, to_value, values):
        if to_value is not None and values.get('from_') is not None:
            if to_value <= values['from_']:
                raise ValueError("'to' value must be greater than 'from' value")
        return to_value

class FieldFacetsConfiguration(StrictBaseModel):
    class Config:
        allow_population_by_field_name = False # disable ability to pass max_results, only alias
    type: Literal["string", "array", "number"]
    order: Optional[Literal["asc", "desc"]] = None
    max_results: Optional[int] = Field(None, alias="maxResults")
    ranges: Optional[List[RangeConfiguration]] = None
    exclude_terms: Optional[List[str]] = Field(None, alias="excludeTerms")

    @validator('max_results')
    def validate_max_results(cls, v):
        if v is not None and v <= 0:
            raise ValueError("'maxResults' must be greater than 0")
        if v is not None and v > 10000:
            raise ValueError("'maxResults' must be less than or equal to 10000")
        return v

    @validator('ranges')
    def validate_ranges_overlap(cls, ranges):
        if ranges:
            # Sort ranges by from_ value
            sorted_ranges = sorted(ranges, key=lambda x: (x.from_ if x.from_ is not None else float('-inf')))

            for i in range(len(sorted_ranges) - 1):
                current = sorted_ranges[i]
                next_range = sorted_ranges[i + 1]

                if (current.to_ is not None and next_range.from_ is not None
                    and current.to_ > next_range.from_):
                    raise ValueError("Range configurations must not overlap")
        return ranges

class FacetsParameters(StrictBaseModel):
    class Config:
        allow_population_by_field_name = False # disable ability to pass max_depth or max_results, only alias
    fields: Dict[str, FieldFacetsConfiguration]
    max_depth: Optional[int] = Field(None, alias="maxDepth")
    max_results: Optional[int] = Field(None, alias="maxResults")
    order: Optional[Literal["asc", "desc"]] = None

    @validator('max_depth')
    def validate_max_depth(cls, v):
        if v is not None and v <= 0:
            raise ValueError("'maxDepth' must be greater than 0")

        return v

    @validator('max_results')
    def validate_max_results(cls, v):
        if v is not None and v <= 0:
            raise ValueError("'maxResults' must be greater than 0")
        if v is not None and v > 10000:
            raise ValueError("'maxResults' must be less than or equal to 10000")
        return v



