from abc import ABC
from enum import Enum
from typing import List, Optional, Any

from pydantic import validator

from marqo.core.models.strict_base_model import StrictBaseModel
from marqo.core.search.search_filter import SearchFilter, MarqoFilterStringParser


class ScoreModifierType(Enum):
    Multiply = 'multiply'
    Add = 'add'


class ScoreModifier(StrictBaseModel):
    field: str
    weight: float
    type: ScoreModifierType


class MarqoQuery(StrictBaseModel, ABC):
    index_name: str
    limit: int
    offset: Optional[int] = None
    searchable_attributes: Optional[List[str]] = None
    attributes_to_retrieve: Optional[List[str]] = None
    filter: Optional[Any] = None
    score_modifiers: Optional[List[ScoreModifier]] = None
    expose_facets: bool = False

    @validator('filter', pre=True, always=True)
    def parse_filter(cls, filter):
        if filter is not None:
            if isinstance(filter, str):
                return MarqoFilterStringParser.parse(filter)
            elif isinstance(filter, SearchFilter):
                return filter
            else:
                raise ValueError(f"filter has to be a string or a SearchFilter, got {type(filter)}")

        return None

    # TODO - add validation to make sure searchable_attributes and attributes_to_retrieve are not empty lists


class MarqoTensorQuery(MarqoQuery):
    vector_query: List[float]
    ef_search: Optional[int] = None
    approximate: bool = True


class MarqoLexicalQuery(MarqoQuery):
    or_phrases: List[str]
    and_phrases: List[str]
    # TODO - validate at least one of or_phrases and and_phrases is not empty


class MarqoHybridQuery(MarqoTensorQuery, MarqoLexicalQuery):
    pass
