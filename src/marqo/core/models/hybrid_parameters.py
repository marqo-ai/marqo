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
    searchable_attributes_lexical: Optional[List[str]] = None
    searchable_attributes_tensor: Optional[List[str]] = None

    # TODO: Figure this out later
    score_modifiers_lexical: Optional[List[ScoreModifier]] = None
    score_modifiers_tensor: Optional[List[ScoreModifier]] = None