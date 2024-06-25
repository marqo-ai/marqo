from abc import ABC
from enum import Enum
from typing import List, Optional

from pydantic import validator, root_validator

from marqo.base_model import StrictBaseModel
from marqo.core.models.score_modifier import ScoreModifier
from marqo.core.search.search_filter import SearchFilter, MarqoFilterStringParser
from marqo.core.models.hybrid_parameters import HybridParameters


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
    hybrid_parameters: HybridParameters

    # Core module will use these fields instead of the score_modifiers_lexical and score_modifiers_tensor inside the HybridParameters
    score_modifiers_lexical: Optional[List[ScoreModifier]] = None
    score_modifiers_tensor: Optional[List[ScoreModifier]] = None
    @root_validator(pre=True)
    def validate_searchable_attributes_and_score_modifiers(cls, values):
        # score_modifiers cannot defined at the same time as score_modifiers_tensor / score_modifiers_lexical
        if values.get("score_modifiers") is not None:
            if values.get("score_modifiers_tensor") is None and values.get("score_modifiers_lexical") is None:
                # Set the score modifiers for both tensor and lexical search to default
                values['score_modifiers_lexical'] = values['score_modifiers']
                values['score_modifiers_tensor'] = values['score_modifiers']
            else:
                raise ValueError("For hybrid search, either define hybrid.parameters.score_modifiers_tensor and "
                                "hybrid.parameters.score_modifiers_lexical or score_modifiers, not both.")

        # searchable_attributes cannot be defined at the same time as searchable_attributes_tensor / searchable_attributes_lexical
        if values.get("searchable_attributes") is not None:
            if values.get("hybrid_parameters") is not None:
                if values.get("hybrid_parameters").searchable_attributes_tensor is None and \
                values.get("hybrid_parameters").searchable_attributes_lexical is None:
                    # Set the searchable attributes for both tensor and lexical search to default
                    values['hybrid_parameters'].searchable_attributes_lexical = values['searchable_attributes']
                    values['hybrid_parameters'].searchable_attributes_tensor = values['searchable_attributes']
                else:
                    raise ValueError("For hybrid search, either define hybrid.parameters.searchable_attributes_tensor and "
                                    "hybrid.parameters.searchable_attributes_lexical or searchable_attributes, not both.")

        return values
