import os
from unittest import mock

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import ScoreModifier
from marqo.vespa.models import QueryResult
from marqo.vespa.models.query_result import Root, Child, RootFields
from tests.marqo_test import MarqoTestCase
from marqo.api.models.update_documents import UpdateDocumentsBodyParams
from marqo.tensor_search.api import update_documents



class TestDictScoreModifiers(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # UNSTRUCTURED indexes
        unstructured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='open_clip/ViT-B-32/laion2b_s34b_b79k')
        )

        # STRUCTURED indexes
        structured_default_text_index = cls.structured_marqo_index_request(
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            vector_numeric_type=VectorNumericType.Float,
            normalize_embeddings=True,
            fields=[
                FieldRequest(name="text_field", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch]),
                FieldRequest(name="map_score_mods", type=FieldType.MapFloat,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="map_score_mods_int", type=FieldType.MapInt,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="map_score_mods_long", type=FieldType.MapLong,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="map_score_mods_double", type=FieldType.MapDouble,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="score_mods_int", type=FieldType.Int,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="score_mods_long", type=FieldType.Long,
                             features=[FieldFeature.ScoreModifier]),
            ],
            hnsw_config=HnswConfig(
                ef_construction=512,
                m=16
            ),
            tensor_fields=["text_field"]
        )

        cls.indexes = cls.create_indexes([
            unstructured_default_text_index,
            structured_default_text_index
        ])

        # Assign to objects so they can be used in tests
        cls.unstructured_default_text_index = cls.indexes[0]
        cls.structured_default_text_index = cls.indexes[1]

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    # Test Add to score
    def test_add_to_score_map_score_modifier(self):
        """
        Test that adding to score works for a map score modifier.
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                            {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                            {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                            {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                            {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                            "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                        mappings={
                            "map_score_mods": {"type": "map_score_modifiers"},
                            "map_score_mods_int": {"type": "map_score_modifiers"}
                        } if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )
                # Search with score modifier
                # 0.5 + 1 * 5 = 5.5
                score_modifier = ScoreModifier(**{"add_to_score": [{"field_name": "map_score_mods_int.c", "weight": 5}]})
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )

                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]

                # Assert that the first result has _id "6" and 5 <= score <= 6
                self.assertIn(res["hits"][0]["_id"], ["6", "7"])
                self.assertTrue(5 <= score_of_first_result <= 6)

    # Test multiply score by
    def test_multiply_score_by_map_score_modifier(self):
        """
        Test that multiplying score by works for a map score modifier.
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                            {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                            {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                            {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                            {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                            "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                        mappings={
                            "map_score_mods": {"type": "map_score_modifiers"},
                            "map_score_mods_int": {"type": "map_score_modifiers"}
                        } if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )
                # Search with score modifier
                # 0.5 * 0.5 * 4 = 1 (1 and 7)
                score_modifier = ScoreModifier(**{"multiply_score_by": [{"field_name": "map_score_mods.a", "weight": 4}]})
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )

                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]

                # Assert that the first result has _id "6" and 0.8 <= score <= 1.2
                self.assertIn(res["hits"][0]["_id"], ["1", "7"])
                self.assertTrue(0.8 <= score_of_first_result <= 1.2)

    # Test combined add to score and multiply score by
    def test_combined_map_score_modifier(self):
        """
        Test that combining adding to score and multiplying score by works for a map score modifier.
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                            {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                            {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                            {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                            {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                            "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                        mappings={
                            "map_score_mods": {"type": "map_score_modifiers"},
                            "map_score_mods_int": {"type": "map_score_modifiers"}
                        } if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )
                # Search with score modifier
                # 0.5 * 0.5 * 4 = 1 (1 and 7)
                score_modifier = ScoreModifier(**{
                        "add_to_score": [{"field_name": "map_score_mods_int.c", "weight": 2}],
                        "multiply_score_by": [{"field_name": "map_score_mods.a", "weight": 4}]
                    }
                )
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )

                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]

                # Assert that the first result has _id "6" and 5 <= score <= 6
                self.assertTrue(res["hits"][0]["_id"] == "7")
                self.assertTrue(2.9 <= score_of_first_result <= 3.1)


    def test_partial_document_update(self):
        """
        Test that partial document update works for a map score modifier.
        """
        for index in [self.structured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods": {"a": 0.5}},
                            {"_id": "2", "text_field": "a photo of a dog", "map_score_mods": {"b": 0.5}},
                            {"_id": "3", "text_field": "a photo of a cat", "map_score_mods": {"c": 0.5}},
                            {"_id": "4", "text_field": "a photo of a cat", "map_score_mods_int": {"a": 1}},
                            {"_id": "5", "text_field": "a photo of a cat", "map_score_mods_int": {"b": 1}},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1}},
                            {"_id": "7", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                            "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                        mappings={
                            "map_score_mods": {"type": "map_score_modifiers"},
                            "map_score_mods_int": {"type": "map_score_modifiers"}
                        } if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Get Document and assert that the score modifier is 0.5
                # Sample:  updated_doc = tensor_search.get_document_by_id(self.config, self.structured_index_name, updated_doc["_id"])
                original_doc = tensor_search.get_document_by_id(self.config, index.name, "1")
                self.assertTrue(original_doc["_id"] == "1")
                self.assertTrue(original_doc["map_score_mods"]["a"] == 0.5)

                # Update the document
                updated_doc = {
                    "map_score_mods": {"a": 1.5},
                    "_id": "1"
                }
                r = update_documents(
                    marqo_config=self.config,
                    index_name=index.name,
                    body=UpdateDocumentsBodyParams(documents=[updated_doc])
                )

                # Get updated document and assert that the score modifier is 1.5
                updated_doc = tensor_search.get_document_by_id(self.config, index.name, "1")
                self.assertTrue(updated_doc["_id"] == "1")
                self.assertTrue(updated_doc["map_score_mods"]["a"] == 1.5)

                # Search with score modifier
                # 0.5 + 1.5 * 2 = 3.5
                score_modifier = ScoreModifier(**{"add_to_score": [{"field_name": "map_score_mods.a", "weight": 2}]})
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )
                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]

                # Assert that the first result has _id "1" and 3 <= score <= 4
                self.assertTrue(res["hits"][0]["_id"] == "1")
                self.assertTrue(3 <= score_of_first_result <= 4)

    def test_long_score_modifier(self):
        """
        Test that long score modifier works for a map score modifier.
        """
        for index in [self.structured_default_text_index, self.unstructured_default_text_index]:
            with self.subTest(index=index.type):
                # Add documents
                res = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=[
                            {"_id": "1", "text_field": "a photo of a cat", "map_score_mods_long": {"a": 4294967295012}},
                            {"_id": "2", "text_field": "a photo of a cat", "score_mods_long": 4294967295012},
                            {"_id": "4", "text_field": "a photo of a cat", "score_mods_long": 1},
                            {"_id": "6", "text_field": "a photo of a cat", "map_score_mods_int": {"c": 1},
                            "map_score_mods": {"a": 0.5}},
                        ],
                        tensor_fields=["text_field"] if isinstance(index, UnstructuredMarqoIndex) else None,
                        mappings={
                            "map_score_mods_long": {"type": "map_score_modifiers"},
                            "map_score_mods_int": {"type": "map_score_modifiers"},
                        } if isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )
                # Search with score modifier
                # 0.5 * 0.5 * 4 = 1 (1 and 7)
                score_modifier = ScoreModifier(**{
                        "add_to_score": [{"field_name": "map_score_mods_long.a", "weight": 20},
                                         {"field_name": "score_mods_long", "weight": 20}],
                    }
                )
                res = tensor_search.search(
                    index_name=index.name, config=self.config, text="",
                    score_modifiers=score_modifier,
                    result_count=10
                )

                # Get the score of the first result.
                score_of_first_result = res["hits"][0]["_score"]
                score_of_second_result = res["hits"][1]["_score"]

                # anticipated score is 4294967295012*20 = 85899345900240
                # Assert that the first and second result both have _ids in 1 and 2
                # and 85899345900239 <= score <= 85899345900241
                self.assertTrue(res["hits"][0]["_id"] in ["2", "1"])
                self.assertTrue(res["hits"][1]["_id"] in ["2", "1"])
                self.assertTrue(85899345900239 <= score_of_first_result <= 85899345900241)
                self.assertTrue(85899345900239 <= score_of_second_result <= 85899345900241)
            
