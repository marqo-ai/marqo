from enum import Enum
from enum import Enum
from typing import List, Optional

from pydantic import validator, root_validator

from marqo.base_model import StrictBaseModel
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists


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
    alpha: Optional[float] = None
    rrf_k: Optional[int] = None
    searchable_attributes_lexical: Optional[List[str]] = None
    searchable_attributes_tensor: Optional[List[str]] = None
    verbose: bool = False

    # Input for API, but form will change before being passed to core Hybrid Query.
    score_modifiers_lexical: Optional[ScoreModifierLists] = None
    score_modifiers_tensor: Optional[ScoreModifierLists] = None

    @root_validator(pre=False)
    def validate_properties(cls, values):
        # alpha can only be defined for RRF and NormalizeLinear
        fusion_ranking_methods = [RankingMethod.RRF, RankingMethod.NormalizeLinear]
        if values.get('alpha') is None:
            if values.get('ranking_method') in fusion_ranking_methods:
                values['alpha'] = 0.5
        else:
            if values.get('ranking_method') not in fusion_ranking_methods:
                raise ValueError(
                    "'alpha' can only be defined for 'rrf' ranking method")  # TODO: Re-add normalize linear

        # rrf_k can only be defined for RRF
        if values.get('rrf_k') is None:
            if values.get('ranking_method') == RankingMethod.RRF:
                values['rrf_k'] = 60
        else:
            if values.get('ranking_method') != RankingMethod.RRF:
                raise ValueError("'rrf_k' can only be defined for 'rrf' ranking method")

        # searchable_attributes_lexical can only be defined for Lexical (ranking or retrieval), Disjunction
        if values.get('searchable_attributes_lexical') is not None:
            if not (values.get('retrieval_method') in [RetrievalMethod.Lexical, RetrievalMethod.Disjunction] or
                    values.get('ranking_method') == RankingMethod.Lexical):
                raise ValueError(
                    "'searchable_attributes_lexical' can only be defined for 'lexical', 'disjunction' retrieval methods or 'lexical' ranking method")

        # searchable_attributes_tensor can only be defined for Tensor (ranking or retrieval), Disjunction
        if values.get('searchable_attributes_tensor') is not None:
            if not (values.get('retrieval_method') not in [RetrievalMethod.Tensor, RetrievalMethod.Disjunction] or
                    values.get('ranking_method') == RankingMethod.Tensor):
                raise ValueError(
                    "'searchable_attributes_tensor' can only be defined for 'tensor', 'disjunction' retrieval methods or 'tensor' ranking method")

        # score_modifiers_lexical can only be defined for Lexical, RRF, NormalizeLinear
        if values.get('score_modifiers_lexical') is not None:
            if values.get('ranking_method') not in [RankingMethod.Lexical, RankingMethod.RRF,
                                                    RankingMethod.NormalizeLinear]:
                raise ValueError(
                    "'score_modifiers_lexical' can only be defined for 'lexical', 'rrf' ranking methods")  # TODO: re-add normalize_linear

        # score_modifiers_tensor can only be defined for Tensor, RRF, NormalizeLinear
        if values.get('score_modifiers_tensor') is not None:
            if values.get('ranking_method') not in [RankingMethod.Tensor, RankingMethod.RRF,
                                                    RankingMethod.NormalizeLinear]:
                raise ValueError(
                    "'score_modifiers_tensor' can only be defined for 'tensor', 'rrf', ranking methods")  # TODO: re-add normalize_linear

        # if retrieval_method == Disjunction, then ranking_method must be RRF, NormalizeLinear
        if values.get('retrieval_method') == RetrievalMethod.Disjunction:
            if values.get('ranking_method') not in [RankingMethod.RRF, RankingMethod.NormalizeLinear]:
                raise ValueError(
                    "For retrieval_method: disjunction, ranking_method must be: rrf")  # TODO: re-add normalize_linear

        # if retrieval_method is Lexical or Tensor, then ranking_method must be Tensor, Lexical
        if values.get('retrieval_method') in [RetrievalMethod.Lexical, RetrievalMethod.Tensor]:
            if values.get('ranking_method') not in [RankingMethod.Lexical, RankingMethod.Tensor]:
                raise ValueError("For retrieval_method: tensor or lexical, ranking_method must be: tensor or lexical")

        return values

    @validator('alpha')
    def validate_alpha(cls, alpha):
        # alpha can only be 0 to 1
        if alpha is not None:
            if alpha < 0 or alpha > 1:
                raise ValueError("alpha can only be between 0 and 1")
        return alpha

    @validator('rrf_k', pre=True)
    def validate_rrf_k(cls, rrf_k):
        # rrf_k can only be int greater than or equal to 0
        if rrf_k is not None:
            if not isinstance(rrf_k, int):
                raise ValueError("rrf_k must be an integer")
            if rrf_k < 0:
                raise ValueError("rrf_k can only be greater than or equal to 0")
        return rrf_k
