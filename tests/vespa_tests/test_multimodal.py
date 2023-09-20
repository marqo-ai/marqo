import os
import json

from marqo.vespa.tensor_search.tensor_search import (create_vector_index, add_documents,
                                                     _vector_text_search, _vespa_lexical_search)
from marqo.config import Config
from marqo.tensor_search.models.api_models import ScoreModifier
from tests.vespa_tests.VespaTestCase import VespaTestCase
from marqo.tensor_search.models.add_docs_objects import AddDocsParams


class TestMultimodal(VespaTestCase):
    """To test the basic functionality of Vespa.
    Please ensure that a Vespa container is running with pre-defined services.xml files
    The content cluster should be named "mind"
    """

    def setUp(self) -> None:
        os.environ["VESPA_CONFIG_URL"] = "http://localhost:19071"
        os.environ["VESPA_QUERY_URL"] = "http://localhost:8080"
        self.config = Config()
        self.index_name = "test_multimodal_index"

        index_settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": "ViT-B/32",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 2,
                    "split_overlap": 0,
                    "split_method": "sentence"
                },
            },

            "documents_structure":
                [
                 {"field_name": "text", "type": "string","indexing_options": ["index", "attribute", "summary"]},
                 {"field_name": "image", "type": "string", "indexing_options": ["index", "attribute", "summary"]},
                 {"field_name": "age", "type": "int", "indexing_options": ["attribute", "summary"]}
                ],
        }
        create_vector_index(config=self.config, index_name=self.index_name, index_settings=index_settings)

        self.document = {
                "_id": "456",
                "multimodal_field": {
                    "text": "A man running on a beach",
                    "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
                },
                "age": 20
            }

        # Delete all docs first
        self.config.vespa_query_client.delete_all_docs(content_cluster_name="mind", schema=self.index_name)

        mappings = {
            "multimodal_field": {
                "type": "multimodal_combination",
                "weights": {
                    "image" : 0.5,
                    "text" : 0.5,
                },
            },
        }

        res = add_documents(config=self.config,
                      add_docs_params=AddDocsParams(index_name=self.index_name, docs=[self.document, ],
                                                    non_tensor_fields=["age"], device="cpu", auto_refresh=True,
                                                    mappings = mappings))

        res = self.get_document_from_index(schema=self.index_name, document_id="456").get_json()

        assert res["fields"]["text"] == self.document["multimodal_field"]["text"]
        assert res["fields"]["image"] == self.document["multimodal_field"]["image"]
        assert len(res["fields"]["marqo_chunks"]) == 1

    def get_document_from_index(self, schema, document_id):
        return self.config.vespa_query_client.get_data(schema=schema, data_id=document_id)

    def test_search(self):
        res = _vector_text_search(config=self.config, index_name=self.index_name, query="A red bus",
                                  device="cpu")
        assert res["hits"][0]["_highlights"]["multimodal_field"] == json.dumps(self.document["multimodal_field"])
