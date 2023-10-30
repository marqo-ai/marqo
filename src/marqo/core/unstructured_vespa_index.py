from typing import Dict, Any

from marqo.core.models import MarqoQuery, MarqoIndex
from marqo.core.vespa_index import VespaIndex


class UnstructuredVespaIndex(VespaIndex):
    @classmethod
    def generate_schema(cls, marqo_index: MarqoIndex) -> str:
        raise NotImplementedError

    @classmethod
    def to_vespa_document(cls, marqo_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def to_marqo_document(cls, vespa_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def to_vespa_query(cls, query: MarqoQuery) -> Dict[str, Any]:
        raise NotImplementedError
