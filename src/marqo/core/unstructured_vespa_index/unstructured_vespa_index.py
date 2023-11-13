from typing import Dict, Any

from marqo.core.models import MarqoQuery
from marqo.core.models.marqo_index import UnstructuredMarqoIndex
from marqo.core.vespa_index import VespaIndex


class UnstructuredVespaIndex(VespaIndex):

    def __init__(self, marqo_index: UnstructuredMarqoIndex):
        self._marqo_index = marqo_index

    def to_vespa_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def to_marqo_document(self, vespa_document: Dict[str, Any], return_highlights: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def to_vespa_query(self, query: MarqoQuery) -> Dict[str, Any]:
        raise NotImplementedError
