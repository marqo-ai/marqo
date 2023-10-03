from abc import ABC, abstractmethod
from typing import Dict, Any

from marqo.core.models import MarqoQuery, MarqoIndex


class VespaIndex(ABC):
    @classmethod
    @abstractmethod
    def generate_schema(cls, marqo_index: MarqoIndex) -> str:
        pass

    @classmethod
    @abstractmethod
    def to_vespa_document(cls, marqo_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def to_marqo_document(cls, vespa_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def to_vespa_query(cls, query: MarqoQuery) -> Dict[str, Any]:
        pass
