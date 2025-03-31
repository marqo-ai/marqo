from enum import Enum
from enum import Enum
from typing import List, Optional, Union

from pydantic import validator, root_validator

from marqo.base_model import StrictBaseModel
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists


class RetrievalMethod(str, Enum):
    Disjunction = 'disjunction'
    Tensor = 'tensor'
    Lexical = 'lexical'


class RankingMethod(str, Enum):
    RRF = 'rrf'
    Tensor = 'tensor'
    Lexical = 'lexical'


class HybridParameters(StrictBaseModel):
    retrievalMethod: Optional[RetrievalMethod] = RetrievalMethod.Disjunction
    rankingMethod: Optional[RankingMethod] = RankingMethod.RRF
    alpha: Optional[float] = None
    rrfK: Optional[int] = None
    searchableAttributesLexical: Optional[List[str]] = None
    searchableAttributesTensor: Optional[List[str]] = None
    verbose: bool = False

    # Input for API, but form will change before being passed to core Hybrid Query.
    scoreModifiersLexical: Optional[ScoreModifierLists] = None
    scoreModifiersTensor: Optional[ScoreModifierLists] = None

    rerankDepthTensor: Optional[int] = None
    queryLexical: Optional[str] = None
    queryTensor: Optional[Union[str, dict]] = None

    @root_validator(pre=False)
    def validate_properties(cls, values):
        # alpha can only be defined for RRF and NormalizeLinear
        fusion_ranking_methods = [RankingMethod.RRF]
        if values.get('alpha') is None:
            if values.get('rankingMethod') in fusion_ranking_methods:
                values['alpha'] = 0.5
        else:
            if values.get('rankingMethod') not in fusion_ranking_methods:
                raise ValueError(
                    "'alpha' can only be defined for 'rrf' ranking method")  # TODO: Re-add normalize linear

        # rrf_k can only be defined for RRF
        if values.get('rrfK') is None:
            if values.get('rankingMethod') == RankingMethod.RRF:
                values['rrfK'] = 60
        else:
            if values.get('rankingMethod') != RankingMethod.RRF:
                raise ValueError("'rrfK' can only be defined for 'rrf' ranking method")

        # searchable_attributes_lexical can only be defined for Lexical (ranking or retrieval), Disjunction
        if values.get('searchableAttributesLexical') is not None:
            if not (values.get('retrievalMethod') in [RetrievalMethod.Lexical, RetrievalMethod.Disjunction] or
                    values.get('rankingMethod') == RankingMethod.Lexical):
                raise ValueError(
                    "'searchableAttributesLexical' can only be defined for 'lexical', 'disjunction' retrieval methods or 'lexical' ranking method")

        # searchable_attributes_tensor can only be defined for Tensor (ranking or retrieval), Disjunction
        if values.get('searchableAttributesTensor') is not None:
            if not (values.get('retrievalMethod') in [RetrievalMethod.Tensor, RetrievalMethod.Disjunction] or
                    values.get('rankingMethod') == RankingMethod.Tensor):
                raise ValueError(
                    "'searchableAttributesTensor' can only be defined for 'tensor', 'disjunction' retrieval methods or 'tensor' ranking method")

        # score_modifiers_lexical can only be defined for Lexical, RRF, NormalizeLinear
        if values.get('scoreModifiersLexical') is not None:
            if not (values.get('rankingMethod') in [RankingMethod.Lexical, RankingMethod.RRF] or
                    values.get('retrievalMethod') == RetrievalMethod.Lexical):
                raise ValueError(
                    "'scoreModifiersLexical' can only be defined for 'lexical', 'rrf' ranking methods or "
                    "'lexical' retrieval method.")  # TODO: re-add normalize_linear

        # score_modifiers_tensor can only be defined for Tensor, RRF, NormalizeLinear
        if values.get('scoreModifiersTensor') is not None:
            if values.get('rankingMethod') not in [RankingMethod.Tensor, RankingMethod.RRF]:
                raise ValueError(
                    "'scoreModifiersTensor' can only be defined for 'tensor', 'rrf', ranking methods")  # TODO: re-add normalize_linear

        # if retrievalMethod == Disjunction, then ranking_method must be RRF, NormalizeLinear
        if values.get('retrievalMethod') == RetrievalMethod.Disjunction:
            if values.get('rankingMethod') not in [RankingMethod.RRF]:
                raise ValueError(
                    "For retrievalMethod: disjunction, rankingMethod must be: rrf")  # TODO: re-add normalize_linear

        # if retrievalMethod is Lexical or Tensor, then ranking_method must be Tensor, Lexical
        if values.get('retrievalMethod') in [RetrievalMethod.Lexical, RetrievalMethod.Tensor]:
            if values.get('rankingMethod') not in [RankingMethod.Lexical, RankingMethod.Tensor]:
                raise ValueError("For retrievalMethod: tensor or lexical, rankingMethod must be: tensor or lexical")

        # if tensor query is an empty dict
        if isinstance(values.get('queryTensor'), dict):
            if not len(values.get('queryTensor')):
                raise ValueError(
                    "Multi-query search for queryTensor requires at least one query! Received empty dictionary. "
                )


        return values

    @validator('alpha')
    def validate_alpha(cls, alpha):
        # alpha can only be 0 to 1
        if alpha is not None:
            if alpha < 0 or alpha > 1:
                raise ValueError("alpha can only be between 0 and 1")
        return alpha

    @validator('rrfK', pre=True)
    def validate_rrf_k(cls, rrfK):
        # rrf_k can only be int greater than or equal to 0
        if rrfK is not None:
            if not isinstance(rrfK, int):
                raise ValueError("rrfK must be an integer")
            if rrfK < 0:
                raise ValueError("rrfK can only be greater than or equal to 0")
        return rrfK
