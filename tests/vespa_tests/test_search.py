import os
import json

from marqo.vespa.tensor_search.tensor_search import (create_vector_index, add_documents,
                                                     _vector_text_search, _vespa_lexical_search)
from marqo.config import Config
from marqo.tensor_search.models.api_models import ScoreModifier
from tests.vespa_tests.VespaTestCase import VespaTestCase
from marqo.tensor_search.models.add_docs_objects import AddDocsParams


class TestVespaBasic(VespaTestCase):
    """To test the basic functionality of Vespa.
    Please ensure that a Vespa container is running with pre-defined services.xml files
    The content cluster should be named "mind"
    """

    def setUp(self) -> None:
        os.environ["VESPA_CONFIG_URL"] = "http://localhost:19071"
        os.environ["VESPA_QUERY_URL"] = "http://localhost:8080"
        self.config = Config()
        self.index_name = "test_index"

        index_settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 2,
                    "split_overlap": 0,
                    "split_method": "sentence"
                },
            },

            "documents_structure":
                [{"field_name": "title", "type": "string", "indexing_options": ["index", "attribute", "summary"],
                  "index_options": ["bm25"]},
                 {"field_name": "description", "type": "string", "indexing_options": ["index", "attribute", "summary"],
                  "index_options": ["bm25"]},
                 # Always need an int or float field to make the score modifier work properly
                 {"field_name": "age", "type": "int", "indexing_options": ["attribute", "summary"]}],
        }
        create_vector_index(config=self.config, index_name=self.index_name, index_settings=index_settings)

        self.document = {
                "_id": "123",
                "title": "A man running on a beach",
                "description": "I like eating apples",
                "age": 20
            }

        # Delete all docs first
        self.config.vespa_query_client.delete_all_docs(content_cluster_name="mind", schema=self.index_name)

        add_documents(config=self.config,
                      add_docs_params=AddDocsParams(index_name=self.index_name, docs=[self.document, ],
                                                    non_tensor_fields=["age"], device="cpu", auto_refresh=True))

        res = self.get_document_from_index(schema=self.index_name, document_id="123").get_json()
        assert res["fields"]["title"] == self.document["title"]
        assert res["fields"]["description"] == self.document["description"]
        assert len(res["fields"]["marqo_chunks"]) == 2

    def get_document_from_index(self, schema, document_id):
        return self.config.vespa_query_client.get_data(schema=schema, data_id=document_id)

    def test_tensor_search(self):
        res = _vector_text_search(config=self.config, index_name=self.index_name, query="A person running along the sea",
                                  device="cpu")["hits"]

        self.assertEqual(1, len(res))

    def test_score_modifier(self):
        weight = 1.2
        regular_score = _vector_text_search(config=self.config, index_name=self.index_name,
                                            query="A person running along the sea",
                                            device="cpu")["hits"][0]["_score"]

        modified_score = _vector_text_search(config=self.config, index_name=self.index_name,
                                             query="A person running along the sea", device="cpu",
                                             score_modifiers= ScoreModifier(**{"multiply_score_by": [{"field_name": "age", "weight": weight}]}))["hits"][0]["_score"]

        assert regular_score * self.document["age"] * weight - modified_score < 1e-5

    def test_lexical_search(self):
        res = _vespa_lexical_search(config=self.config, index_name=self.index_name,
                                    text="A person running along the sea")["hits"]
        assert len(res) == 1

        bad_res = _vespa_lexical_search(config=self.config, index_name=self.index_name, text="marqo test")["hits"]
        assert len(bad_res) == 0

    def test_hybrid_search(self):
        res = _vector_text_search(config=self.config, index_name=self.index_name, query="A person running along the sea",
                                  searchable_attributes=["::hybrid"], device="cpu")["hits"]
        assert len(res) == 1

    def test_exact_knn_search(self):
        res = _vector_text_search(config=self.config, index_name=self.index_name, query="A person running along the sea",
                                  searchable_attributes=["::exact_knn"], device="cpu")["hits"]
        assert len(res) == 1
