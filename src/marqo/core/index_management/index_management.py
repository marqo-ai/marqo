from marqo.core.models import MarqoIndex
from marqo.vespa.vespa_client import VespaClient


class IndexManagement:
    def __init__(self, vespa_client: VespaClient):
        self.vespa_client = vespa_client

    def create_index(self, marqo_index: MarqoIndex):
        pass
