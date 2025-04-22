from abc import ABC
from enum import Enum
from typing import List, Optional

from pydantic import validator, root_validator

from marqo.base_model import StrictBaseModel
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.score_modifier import ScoreModifier
from marqo.core.search.search_filter import SearchFilter, MarqoFilterStringParser
from marqo.core.models.hybrid_parameters import RankingMethod, HybridParameters


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
    rerank_depth_tensor: Optional[int] = None

    # TODO - validate that ef_search >= offset+limit if provided


class MarqoLexicalQuery(MarqoQuery):
    or_phrases: List[str]
    and_phrases: List[str]

    # Both lists can be empty only if it's a MarqoHybridQuery and it's
    # retrieval_method & ranking_method are "TENSOR" (i.e. it's a pure tensor search)


class MarqoHybridQuery(MarqoTensorQuery, MarqoLexicalQuery):
    hybrid_parameters: HybridParameters
    vector_query: Optional[List[float]] # overrides tensor parameter to allow None value.

    # Core module will use these fields instead of the score_modifiers_lexical and score_modifiers_tensor inside the HybridParameters
    score_modifiers_lexical: Optional[List[ScoreModifier]] = None
    score_modifiers_tensor: Optional[List[ScoreModifier]] = None
    global_rerank_depth: Optional[int] = None
    facets: Optional[FacetsParameters] = None
    track_total_hits: Optional[bool] = None

    @root_validator(pre=True)
    def validate_searchable_attributes_and_score_modifiers(cls, values):
        # score_modifiers can only be set for hybrid search - RRF
        hybrid_parameters = values.get("hybrid_parameters")
        if values.get("score_modifiers") is not None and hybrid_parameters.rankingMethod != RankingMethod.RRF:
            raise ValueError(f"'scoreModifiers' is only supported for hybrid search if 'rankingMethod' is 'RRF'. "
                             f"For your 'rankingMethod': {hybrid_parameters.rankingMethod}, define the "
                             f"'scoreModifiersTensor' and/or 'scoreModifiersLexical' keys inside the "
                             f"'hybridParameters' dict parameter.")

        # searchable_attributes cannot be defined for hybrid search
        if values.get("searchable_attributes") is not None:
            raise ValueError("'searchableAttributes' cannot be used for hybrid search. Instead, define the "
                             "'searchableAttributesTensor' and/or 'searchableAttributesLexical' keys inside the "
                             "'hybridParameters' dict parameter.")

        return values
