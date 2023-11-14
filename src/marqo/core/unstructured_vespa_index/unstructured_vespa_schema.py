from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.core.vespa_schema import VespaSchema


class UnstructuredVespaSchema(VespaSchema):
    def __init__(self, marqo_index: UnstructuredMarqoIndexRequest):
        self.marqo_index = marqo_index

    def generate_schema(self) -> (str, MarqoIndex):
        pass
