from abc import ABC
from enum import Enum
from typing import List, Optional

from pydantic import validator, BaseModel, root_validator

from marqo.base_model import StrictBaseModel
from marqo.core.search.search_filter import SearchFilter, MarqoFilterStringParser
from marqo.core.models.score_modifier import ScoreModifier


class RetrievalMethod(str, Enum):
    Disjunction = 'disjunction'
    Tensor = 'tensor'
    Lexical = 'lexical'

class RankingMethod(str, Enum):
    RRF = 'rrf'
    NormalizeLinear = 'normalize_linear'
    Tensor = 'tensor'
    Lexical = 'lexical'


class HybridParameters(StrictBaseModel):
    retrieval_method: Optional[RetrievalMethod] = RetrievalMethod.Disjunction
    ranking_method: Optional[RankingMethod] = RankingMethod.RRF
    alpha: Optional[float] = 0.5
    rrf_k: Optional[int] = 60
    searchable_attributes_lexical: Optional[List[str]] = None
    searchable_attributes_tensor: Optional[List[str]] = None
    verbose: bool = False

    # TODO: Figure this out later
    score_modifiers_lexical: Optional[List[ScoreModifier]] = None
    score_modifiers_tensor: Optional[List[ScoreModifier]] = None

    @root_validator(pre=True)
    def validate_properties(cls, values):
        # alpha can only be defined for RRF and NormalizeLinear
        # rrf_k can only be defined for RRF
        # searchable_attributes_lexical can only be defined for Lexical, RRF, NormalizeLinear
        # searchable_attributes_tensor can only be defined for Tensor, RRF, NormalizeLinear
        # score_modifiers_lexical can only be defined for Lexical, RRF, NormalizeLinear
        # score_modifiers_tensor can only be defined for Tensor, RRF, NormalizeLinear

        # if retrieval_method == Disjunction, then ranking_method must be RRF, NormalizeLinear
        # if retrieval_method == Tensor, then ranking_method must be Tensor, Lexical
        # if retrieval_method == Lexical, then ranking_method must be Tensor, Lexical

        return values

    # alpha can only be 0 to 1
    # rrf_k can only be int greater than or equal to 0