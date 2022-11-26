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
                    "Description": "A photo of a hippo",
                    "Image location": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
                }
            ],
            enrichment={
                "model": "attribute-extraction",
                "to": ["Is_bathroom", "Is_Bedroom", "Is_Study", "Is_Yard", "Is_Hippo"],
                "kwargs": {
                    "attributes": [{"string": "Bathroom, Bedroom, Study, Yard, Hippo"}],
                    "image_field": {"document_field": "Image location"}
                },
            },
            indexing_instructions=[],
            device='cpu'
        )
        print('resultresult')
        pprint.pprint(result)