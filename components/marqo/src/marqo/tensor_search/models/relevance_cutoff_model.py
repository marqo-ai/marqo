from enum import Enum
from typing import Optional, Union

from pydantic.v1 import root_validator, Field

from marqo.base_model import StrictBaseModel
from marqo.core.models.hybrid_parameters import LexicalOperand


class ApplyInRetrieval(str, Enum):
    Lexical = 'lexical'
    Tensor = 'tensor'
    Both = 'both'


class RelevanceCutoffMethod(str, Enum):
    RelativeMaxScore = "relative_max_score"
    GapDetection = "gap_detection"
    MeanStdDev = "mean_std_dev"


class RelativeMaxScoreParameters(StrictBaseModel):
    relative_score_factor: float = Field(..., ge=0, le=1, alias="relativeScoreFactor")


class MeanStdParameters(StrictBaseModel):
    std_dev_factor: float = Field(..., alias="stdDevFactor")


class RelevanceCutoffModel(StrictBaseModel):
    """
    The RelevanceCutoffModel defines how to apply relevance cutoff in search results.

    Attributes:
        method (RelevanceCutoffMethod): The method to use for relevance cutoff.
        probe_depth (int): The number of documents to probe for relevance cutoff. Defaults to 1000. We use
            a lexical search as a probe search. Check Vespa Custom Searcher for more details.
        parameters (Union[RelativeMaxScoreParameters, MeanStdParameters]): The parameters for the relevance cutoff method.
            If the method is RelativeMaxScore, you must provide 'relativeScoreFactor' as a parameter.
            If the method is MeanStd, you must provide 'stdDevFactor' as a parameter.
            Check Vespa Custom Searcher for more details.
        affect_facets (bool): When True, facets and totalHits will only count documents that pass the
            relevance cutoff. Defaults to False.
    """
    class Config(StrictBaseModel.Config):
        use_enum_values = True

    method: RelevanceCutoffMethod
    probe_depth: int = Field(1000, ge=1, alias="probeDepth")
    parameters: Union[RelativeMaxScoreParameters, MeanStdParameters, None] = None
    affect_facets: bool = Field(False, alias="affectFacets")
    override_sort_candidates_with_relevant_candidates: bool = Field(
        False, alias="overrideSortCandidatesWithRelevantCandidates"
    )
    lexical_operand: Optional[LexicalOperand] = Field(None, alias="lexicalOperand")
    apply_in_retrieval: Optional[ApplyInRetrieval] = Field(None, alias="applyInRetrieval")
    override_total_hits_with_post_process_candidates: bool = Field(
        False, alias="overrideTotalHitsWithPostProcessCandidates"
    )
    override_limit_plus_offset: bool = Field(
        False, alias="overrideLimitPlusOffset"
    )

    @root_validator(pre=False, skip_on_failure=True)
    def _validate_apply_in_retrieval_lexical_not_supported(cls, values):
        apply_in_retrieval = values.get('apply_in_retrieval')
        if apply_in_retrieval == ApplyInRetrieval.Lexical:
            raise ValueError(
                "applyInRetrieval='lexical' is not currently supported. "
                "Only 'tensor' and 'both' are available."
            )
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def _validate_apply_in_retrieval_incompatible_with_override_sort_candidates(cls, values):
        apply_in_retrieval = values.get('apply_in_retrieval')
        override_sort = values.get('override_sort_candidates_with_relevant_candidates')
        if (apply_in_retrieval is not None
                and apply_in_retrieval != ApplyInRetrieval.Both
                and override_sort):
            raise ValueError(
                "applyInRetrieval cannot be used together with "
                "overrideSortCandidatesWithRelevantCandidates when targeting a specific "
                "retrieval leg. relevantCandidates only reflects one leg's cutoff and "
                "would incorrectly trim the combined sort pool."
            )
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def _validate_method_and_parameters(cls, values):
        """
        Validates that the parameters provided match the method selected for relevance cutoff.
        """
        method = values.get('method')
        parameters = values.get('parameters')

        if method == RelevanceCutoffMethod.RelativeMaxScore:
            if not isinstance(parameters, RelativeMaxScoreParameters):
                raise ValueError(f"You must provide '{[f.alias for f in RelativeMaxScoreParameters.__fields__.values()]}'"
                                 f" as parameters for method '{method}'")
        elif method == RelevanceCutoffMethod.MeanStdDev:
            if not isinstance(parameters, MeanStdParameters):
                raise ValueError(f"You must provide '{[f.alias for f in MeanStdParameters.__fields__.values()]}'"
                                 f" as parameters for {method}")
        elif method == RelevanceCutoffMethod.GapDetection:
            if parameters is not None:
                raise ValueError(f"{method} does not require any parameters, but received {parameters}")
        else:
            raise ValueError(f"Unknown relevance cutoff method: {method}")
        return values