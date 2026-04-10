from enum import Enum
from typing import List, Optional, Union

from pydantic.v1 import validator, root_validator, Field

from marqo.base_model import StrictBaseModel
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists


class LexicalOperand(str, Enum):
    Or = 'or'
    And = 'and'
    WeakAnd = 'weakAnd'


class RetrievalMethod(str, Enum):
    Disjunction = 'disjunction'
    Tensor = 'tensor'
    Lexical = 'lexical'


class RankingMethod(str, Enum):
    RRF = 'rrf'
    Tensor = 'tensor'
    Lexical = 'lexical'


class WeakAndParameters(StrictBaseModel):
    stopwordLimit: Optional[float] = Field(None, ge=0, le=1)
    adjustTarget: Optional[float] = Field(None, ge=0, le=1)
    allowDropAll: Optional[bool] = None
    filterThreshold: Optional[float] = Field(None, ge=0, le=1)

    def convert_to_vespa_query_dict(self):
        dict = {
            "ranking.matching.weakand.stopwordLimit": self.stopwordLimit,
            "ranking.matching.weakand.adjustTarget": self.adjustTarget,
            "ranking.matching.weakand.allowDropAll": self.allowDropAll,
            "ranking.matching.filterThreshold": self.filterThreshold,
        }
        return {k: v for k, v in dict.items() if v is not None}


class HybridParameters(StrictBaseModel):
    class Config(StrictBaseModel.Config):
        use_enum_values = True

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
    rerankDepthLexical: Optional[int] = Field(None, ge=1)

    queryLexical: Optional[str] = None
    queryTensor: Optional[Union[str, dict]] = None

    weakAndParameters: Optional[WeakAndParameters] = None
    rerankCount: Optional[int] = Field(None, ge=1)
    secondPhaseModifier: Optional[bool] = None
    lexicalOperand: Optional[LexicalOperand] = None

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

    @root_validator(pre=False)
    def validate_and_set_rerankDepthLexical(cls, values):
        # We do not distinguish between default None and explicitly provided None here
        rerank_depth_lexical = values.get('rerankDepthLexical')
        retrieval_method = values.get('retrievalMethod')

        if rerank_depth_lexical is not None and retrieval_method not in [RetrievalMethod.Lexical,
                                                                         RetrievalMethod.Disjunction]:
            raise ValueError(
                "'rerankDepthLexical' can only be set when 'retrievalMethod' is 'lexical' or 'disjunction'"
            )
        return values

    @root_validator(pre=False)
    def validate_weakand_parameters(cls, values):
        rerank_depth_lexical = values.get('rerankDepthLexical')
        weak_and_parameters = values.get('weakAndParameters')

        if rerank_depth_lexical is None and weak_and_parameters is not None:
            raise ValueError(
                "'weakAndParameters' can only be set when 'rerankDepthLexical' is set"
            )
        return values

    @root_validator(pre=False)
    def validate_second_phase_modifier(cls, values):
        second_phase_modifier = values.get('secondPhaseModifier')
        ranking_method = values.get('rankingMethod')
        retrieval_method = values.get('retrievalMethod')

        if second_phase_modifier is True and not (
            (retrieval_method == RetrievalMethod.Lexical and ranking_method == RankingMethod.Lexical)
            or retrieval_method == RetrievalMethod.Disjunction
        ):
            raise ValueError(
                "'secondPhaseModifier' can only be set to True when 'retrievalMethod' is 'disjunction' or both "
                "'retrievalMethod' and 'rankingMethod' are 'lexical'"
            )
        return values