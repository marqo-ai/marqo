import pprint

from marqo.tensor_search import enrichment
from marqo.errors import ArgParseError
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase


class TestEnrichment(MarqoTestCase):

    def test_enrich(self):
        result = enrichment.enrich(
            documents=[
                {
                    "Description": "A photo of a house",
                    "Image location": "https://s3.image.png"
                }
            ],
            enrichment={
                "model": "attribute-extraction",
                "to": ["Is_bathroom", "Is_Bedroom", "Is_Study", "Is_Yard"],
                "kwargs": {
                    "attributes": [{"string": "Bathroom, Bedroom, Study, Yard"}],
                    "image_field": {"document_field": "Image location"}
                },
            },
            indexing_instructions=[],
            device='cpu'
        )
        pprint.pprint(result)