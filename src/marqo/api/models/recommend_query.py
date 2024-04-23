from typing import Dict, List, Union, Optional

from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.tensor_search.models.api_models import BaseMarqoModel
from marqo.tensor_search.models.score_modifiers_object import ScoreModifier


class RecommendQuery(BaseMarqoModel):
    documents: Union[List[str], Dict[str, float]]
    tensorFields: Optional[List[str]] = None
    interpolationMethod: Optional[InterpolationMethod] = None
    excludeInputDocuments: bool = True
    limit: int = 10
    offset: int = 0
    efSearch: Optional[int] = None
    approximate: Optional[bool] = None
    searchableAttributes: Optional[List[str]] = None
    showHighlights: bool = True
    reRanker: str = None
    filter: str = None
    attributesToRetrieve: Union[None, List[str]] = None
    scoreModifiers: Optional[ScoreModifier] = None
