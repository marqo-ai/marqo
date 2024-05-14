from abc import ABC
from enum import Enum
from typing import List, Optional

from pydantic import validator, BaseModel, root_validator

from marqo.base_model import StrictBaseModel
from marqo.core.search.search_filter import SearchFilter, MarqoFilterStringParser


class RetrievalMethod(str, Enum):
    Disjunction = 'disjunction'
    TensorFirst = 'tensor_first'
    LexicalFirst = 'lexical_first'


class FusionMethod(str, Enum):
    RRF = 'rrf'
    NormalizeLinear = 'normalize_linear'


class HybridParameters(StrictBaseModel):
    retrieval_method: Optional[RetrievalMethod] = RetrievalMethod.Disjunction
    fusion_method: Optional[FusionMethod] = FusionMethod.RRF
    alpha: Optional[float] = 0.5
    k: Optional[int] = 60
    searchable_attributes_lexical: Optional[List[str]] = None
    searchable_attributes_tensor: Optional[List[str]] = None

    # TODO: Figure this out later
    score_modifiers_lexical: Optional[List[ScoreModifier]] = None
    score_modifiers_tensor: Optional[List[ScoreModifier]] = None