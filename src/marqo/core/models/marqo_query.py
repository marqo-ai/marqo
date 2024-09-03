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
    rerank_count: Optional[int] = None
    ef_search: Optional[int] = None
    approximate: bool = True

    @root_validator(pre=True)
    def validate_ef_search_and_rerank_count(cls, values):
        # ef_search must be greater than or equal to limit + offset if provided
        ef_search = values.get("ef_search")
        limit = values.get("limit")
        offset = values.get("offset", 0)
        if ef_search is not None:
            if ef_search < limit + offset:
                raise ValueError(f"Your 'ef_search' is too low. For a tensor search, 'efSearch' must be greater than "
                                 f"or equal to 'limit' + 'offset'. Your current efSearch is {ef_search}, limit is "
                                 f"{limit}, offset is {offset}.")

        # rerank_count must be between limit + offset and ef_search (inclusive) if provided
        rerank_count = values.get("rerank_count")
        if rerank_count is not None:
            if rerank_count < limit + offset:
                raise ValueError(f"Your 'rerank_count' is too low. For a tensor search, 'rerankCount' must be greater "
                                 f"than or equal to 'limit' + 'offset'. Your current rerankCount is {rerank_count}, "
                                 f"limit is {limit}, offset is {offset}.")
            if rerank_count > ef_search:
                raise ValueError(f"Your 'rerank_count' is too high. For a tensor search, 'rerankCount' must be less "
                                 f"than or equal to 'efSearch'. Your current rerankCount is {rerank_count}, efSearch "
                                 f"is {ef_search}.")
        return values


class MarqoLexicalQuery(MarqoQuery):
    or_phrases: List[str]
    and_phrases: List[str]

    # Both lists can be empty only if it's a MarqoHybridQuery and it's
    # retrieval_method & ranking_method are "TENSOR" (i.e. it's a pure tensor search)


class MarqoHybridQuery(MarqoTensorQuery, MarqoLexicalQuery):
    hybrid_parameters: HybridParameters

    # Core module will use these fields instead of the score_modifiers_lexical and score_modifiers_tensor inside the HybridParameters
    score_modifiers_lexical: Optional[List[ScoreModifier]] = None
    score_modifiers_tensor: Optional[List[ScoreModifier]] = None
    @root_validator(pre=True)
    def validate_searchable_attributes_and_score_modifiers(cls, values):
        # score_modifiers cannot defined for hybrid search
        if values.get("score_modifiers") is not None:
            raise ValueError("'scoreModifiers' cannot be used for hybrid search. Instead, define the "
                             "'scoreModifiersTensor' and/or 'scoreModifiersLexical' keys inside the "
                             "'hybridParameters' dict parameter.")

        # searchable_attributes cannot be defined for hybrid search
        if values.get("searchable_attributes") is not None:
            raise ValueError("'searchableAttributes' cannot be used for hybrid search. Instead, define the "
                             "'searchableAttributesTensor' and/or 'searchableAttributesLexical' keys inside the "
                             "'hybridParameters' dict parameter.")

        return values
