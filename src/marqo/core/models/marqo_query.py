from abc import ABC
from enum import Enum
from typing import List, Optional

from pydantic import validator

from marqo.base_model import StrictBaseModel
from marqo.core.search.search_filter import SearchFilter, MarqoFilterStringParser


class ScoreModifierType(Enum):
    Multiply = 'multiply'
    Add = 'add'


class ScoreModifier(StrictBaseModel):
    field: str
    weight: float
    type: ScoreModifierType


class MarqoQuery(StrictBaseModel, ABC):
    class Config(StrictBaseModel.Config):
        arbitrary_types_allowed = True  # To allow SearchFilter

    index_name: str
    limit: int
    offset: Optional[int] = None
    searchable_attributes: Optional[List[str]] = None
    attributes_to_retrieve: Optional[List[str]] = None
    filter: Optional[SearchFilter] = None
    score_modifiers: Optional[List[ScoreModifier]] = None
    expose_facets: bool = False

    @validator('filter', pre=True, always=True)
    def parse_filter(cls, filter):
        if filter is not None:
            if isinstance(filter, str):
                parser = MarqoFilterStringParser()
                return parser.parse(filter)
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

    # TODO - validate that ef_search >= offset+limit if provided


class MarqoLexicalQuery(MarqoQuery):
    or_phrases: List[str]
    and_phrases: List[str]
    # TODO - validate at least one of or_phrases and and_phrases is not empty


class MarqoHybridQuery(MarqoTensorQuery, MarqoLexicalQuery):
    pass
