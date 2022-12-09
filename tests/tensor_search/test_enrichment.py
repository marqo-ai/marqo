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
                },
                {
                    "Description": "A photo of a hippo status",
                    "Image location": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"
                }
            ],
            enrichment={
                "task": "attribute-extraction",
                "to": ["Is_bathroom", "Is_Bedroom", "Is_Office", "Is_Yard", "Is_Hippo"],
                "kwargs": {
                    "attributes": [{"string": "Bathroom, Bedroom, Office, Yard, Hippo"}],
                    "image_field": {"document_field": "Image location"}
                },
            },
            indexing_instructions=[],
            device='cpu'
        )
        assert len(result['results']) == len(result['documents'])
        assert len(result['results']) == 2
        for doc in result['documents']:
            assert "Description" in doc
            assert "Image location" in doc
            for to_field in ["Is_bathroom", "Is_Bedroom", "Is_Office", "Is_Yard", "Is_Hippo"]:
                assert to_field in doc
            assert doc["Is_Hippo"]

        for res in result['results']:
            assert "result" in res
