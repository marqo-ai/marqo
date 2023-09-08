import copy
from unittest import mock
from tests.utils.transition import add_docs_caller
from marqo.errors import IndexNotFoundError, InvalidArgError, BackendCommunicationError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.api_models import ScoreModifier
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity
import pprint
from marqo.tensor_search.tensor_search import (_create_normal_tensor_search_query, _vector_text_search_query_verbose,
                                                _generate_vector_text_search_query_for_verbose_one,
                                               _create_score_modifiers_tensor_search_query)
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierOperator
import numpy as np

from pydantic.error_wrappers import ValidationError
import os

class TestScoreModifiersSearch(MarqoTestCase):

    def setUp(self):
        self.index_name = "test-index"

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except:
            pass

        tensor_search.create_vector_index(
            index_name=self.index_name, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: True
                }
            })
        pass

        self.test_valid_score_modifiers_list = [ScoreModifier(**x) for x in [
            {
                # miss one weight
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1,},
                     {"field_name": "multiply_2",}],
                "add_to_score": [
                    {"field_name": "add_1", "weight" : -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # 0 weight
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1.0, },
                     {"field_name": "multiply_2",
                      "weight": 0}],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # remove one field
                "multiply_score_by":
                    [
                     {"field_name": "multiply_2",
                      "weight": 1.2}],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # missing "multipy_score_by"
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # missing "add_to_score"
                "multiply_score_by":
                    [{"field_name": "multiply_2",
                      "weight": 1.2}],
            },
        ]]

        self.test_score_documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 4 fields
             "multiply_1": 1,
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 30.0,
             "_id": "1"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # let's multiply by 0
             "multiply_1": 0,
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 3.0,
             "_id": "2"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 2 fields
             "multiply_2": 20.3,
             "add_1": 1.2,
             "_id": "3"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # change order
             "add_1": 1.0,
             "add_2": 3.0,
             "multiply_1": 1,
             "multiply_2": -20.0,
             "_id": "4"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # float
             "multiply_1": 1,
             "multiply_2": 2.531,
             "add_1": 1.454,
             "add_2": -3.692,
             "_id": "5"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]

        # Any tests that call add_document, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()
        
    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except:
            pass
        self.device_patcher.stop()

    def test_search_result_not_affected_if_fields_not_exist(self):
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             "_id": "0",
             "filter": "original"
            },
        ]

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)


        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                             text = "what is the rider doing?", score_modifiers=None, result_count=10)

        normal_score = normal_res["hits"][0]["_score"]

        modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                text = "what is the rider doing?",
                                                score_modifiers=ScoreModifier(**{
                                                    "multiply_score_by":
                                                        [{"field_name": "multiply_1",
                                                          "weight": 1,},
                                                         {"field_name": "multiply_2",}
                                                        ],
                                                    "add_to_score": [
                                                        {"field_name": "add_1",
                                                         },
                                                        {"field_name": "add_2",
                                                         }
                                                    ]
                                                }))

        modifier_score = modifier_res["hits"][0]["_score"]
        self.assertEqual(normal_score, modifier_score)

    def get_expected_score(self, doc, ori_score, score_modifiers: ScoreModifier):
        add = 0.0
        if score_modifiers.multiply_score_by is not None:
            for config in score_modifiers.multiply_score_by:
                if config.field_name in doc:
                    if isinstance(doc[config.field_name], (int, float)):
                        ori_score = ori_score * config.weight * doc[config.field_name]

        if score_modifiers.add_to_score is not None:
            for config in score_modifiers.add_to_score:
                if config.field_name in doc:
                    if isinstance(doc[config.field_name], (int, float)):
                        add = add + config.weight * doc[config.field_name]

        return max(0.0, (ori_score + add))

    def test_search_score_modified_as_expected(self):
        documents = self.test_score_documents

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10)
        normal_score = normal_res["hits"][0]["_score"]


        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                    text = "what is the rider doing?",
                                                    score_modifiers=score_modifiers, result_count=10)

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError
                
            tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(index=self.index_name, q="what is the rider doing?", limit=10, scoreModifiers=score_modifiers),
                ]
            ))

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_search_score_modified_as_expected_with_filter(self):
        documents = self.test_score_documents

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10)
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                text="what is the rider doing?",
                                                score_modifiers=score_modifiers, result_count=10,
                                                filter="filter:original")

            assert len(modifier_res["hits"]) == 1
            assert modifier_res["hits"][0]["_id"] == "0"
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_search_score_modified_as_expected_with_searchable_attributes(self):
        documents = self.test_score_documents

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10,
                                          searchable_attributes=["my_image_field"])
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                text="what is the rider doing?",
                                                score_modifiers=score_modifiers, result_count=10,
                                                searchable_attributes=["my_image_field"])

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_search_score_modified_as_expected_with_attributes_to_retrieve(self):
        documents = self.test_score_documents
        invalid_fields = copy.deepcopy(documents[0])
        del invalid_fields["_id"]

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10,)
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                text="what is the rider doing?",
                                                score_modifiers=score_modifiers, result_count=10,
                                                attributes_to_retrieve=["_id"])

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                for field in list(invalid_fields):
                    assert field not in result
                # need to get the original doc with score modifier fields to compute expected score
                original_doc = [doc for doc in documents if doc["_id"] == result["_id"]][0]
                expected_score = self.get_expected_score(original_doc, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_search_score_modified_as_expected_with_skipped_fields(self):
        # multiply_1 is string here, which should be skipped in modification
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 4 fields
             "multiply_1": "1",
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 30.0,
             "_id": "1"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # let's multiply by 0
             "multiply_1": "0",
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 3.0,
             "_id": "2"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 2 fields
             "multiply_2": 20.3,
             "add_1": 1.2,
             "_id": "3"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # change order
             "add_1": 1.0,
             "add_2": 3.0,
             "multiply_1": "1",
             "multiply_2": -20.0,
             "_id": "4"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # float
             "multiply_1": "1",
             "multiply_2": 2.531,
             "add_1": 1.454,
             "add_2": -3.692,
             "_id": "5"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)

        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10)
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                                    text = "what is the rider doing?",
                                                    score_modifiers=score_modifiers, result_count=10)

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_expected_error_raised(self):
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 4 fields
             "multiply_1": 1,
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 30.0,
             "_id": "1"
             },
        ]

        invalid_score_modifiers_list = [
            {
                # typo in multiply_score_by
                "multiply_scores_by":
                    [{"field_name": "reputation",
                      "weight": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],
                "add_to_score": [
                    {"field_name": "rate",
                     }],
            },
            {
                # typo in add_to_score
                "multiply_score_by":
                    [{"field_name": "reputation",
                      "weight": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],
                "add_ssto_score": [
                    {"field_name": "rate",
                     }],
            },
            {
                # typo in field_name
                "multiply_score_by":
                    [{"field_names": "reputation",
                      "weight": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],
                "add_to_score": [
                    {"field_name": "rate",
                     }],
            },
            {
                # typo in weight
                "multiply_score_by":
                    [{"field_names": "reputation",
                      "weight": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],
                "add_to_score": [
                    {"field_name": "rate",
                     }],
            },
            {
                # no field name
                "multiply_scores_by":
                    [{"field_names": "reputation",
                      "weights": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],
                "add_ssto_score": [
                    {"field_name": "rate",
                     }],
            },
            {
                # list in field_name value
                "multiply_score_by":
                    [{"field_name": ["repuation", "reputation-test"],
                      "weight": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     },],
                "add_to_score": [
                    {"field_name": "rate",
                     }]
            },
            {
                # field name can't be "_id"
                "multiply_score_by":
                    [{"field_name": "_id",
                      "weight": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],

                "add_to_score": [
                    {"field_name": "rate",
                     }]
            },
            { # empty
            },
            {  # one part to be empty
                "multiply_score_by": [],
                "add_to_score": [
                    {"field_name": "rate",
                     }]
            },
            {  # two parts to be empty
                "multiply_score_by": [],
                "add_to_score": [],
            },
        ]

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        
        for invalid_score_modifiers in invalid_score_modifiers_list:
            try:
                v = ScoreModifier(**invalid_score_modifiers)
                raise AssertionError(invalid_score_modifiers, v)
            except InvalidArgError:
                pass

    def test_normal_query_body_is_called(self):
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]
        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        def pass_create_normal_tensor_search_query(*arg, **kwargs):
            return _create_normal_tensor_search_query(*arg, **kwargs)

        mock_create_normal_tensor_search_query = mock.MagicMock()
        mock_create_normal_tensor_search_query.side_effect = pass_create_normal_tensor_search_query

        @mock.patch("marqo.tensor_search.tensor_search._create_normal_tensor_search_query", mock_create_normal_tensor_search_query)
        def run():
            tensor_search.search(config=self.config, index_name=self.index_name,
                                              text="what is the rider doing?", score_modifiers=None, result_count=10)
            mock_create_normal_tensor_search_query.assert_called()

            return True
        assert run()


class TestScoreModifiersBulkSearch(MarqoTestCase):

    def setUp(self):
        self.index_name = "test-index"

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except:
            pass

        tensor_search.create_vector_index(
            index_name=self.index_name, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: True
                }
            })
        pass

        self.test_valid_score_modifiers_list = [ScoreModifier(**x) for x in [
            {
                # miss one weight
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1,},
                     {"field_name": "multiply_2",}],
                "add_to_score": [
                    {"field_name": "add_1", "weight" : -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # 0 weight
                "multiply_score_by":
                    [{"field_name": "multiply_1",
                      "weight": 1.0, },
                     {"field_name": "multiply_2",
                      "weight": 0}],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # remove one field
                "multiply_score_by":
                    [
                     {"field_name": "multiply_2",
                      "weight": 1.2}],
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # missing "multipy_score_by"
                "add_to_score": [
                    {"field_name": "add_1", "weight": -3,
                     },
                    {"field_name": "add_2", "weight": 1,
                     }]
            },
            {
                # missing "add_to_score"
                "multiply_score_by":
                    [{"field_name": "multiply_2",
                      "weight": 1.2}],
            },
        ]]

        self.test_score_documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 4 fields
             "multiply_1": 1,
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 30.0,
             "_id": "1"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # let's multiply by 0
             "multiply_1": 0,
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 3.0,
             "_id": "2"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 2 fields
             "multiply_2": 20.3,
             "add_1": 1.2,
             "_id": "3"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # change order
             "add_1": 1.0,
             "add_2": 3.0,
             "multiply_1": 1,
             "multiply_2": -20.0,
             "_id": "4"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # float
             "multiply_1": 1,
             "multiply_2": 2.531,
             "add_1": 1.454,
             "add_2": -3.692,
             "_id": "5"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]

        # Any tests that call add_document, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()
        
    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name)
        except:
            pass
        self.device_patcher.stop()
    
    def test_bulk_search_result_not_affected_if_fields_not_exist(self):
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
            "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
            "_id": "0",
            "filter": "original"
            },
        ]

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                    "filter"], auto_refresh=True)

        bulk_search_query = BulkSearchQuery(
            queries=[
                BulkSearchQueryEntity(
                    index=self.index_name,
                    q="what is the rider doing?",
                    scoreModifiers=None
                )
            ]
        )
        
        normal_res = tensor_search.bulk_search(marqo_config=self.config, query=bulk_search_query)
        normal_score = normal_res["result"][0]["hits"][0]["_score"]

        bulk_search_query_with_modifier = BulkSearchQuery(
            queries=[
                BulkSearchQueryEntity(
                    index=self.index_name,
                    q="what is the rider doing?",
                    scoreModifiers=ScoreModifier(**{
                        "multiply_score_by":
                            [{"field_name": "multiply_1",
                            "weight": 1,},
                            {"field_name": "multiply_2",}
                            ],
                        "add_to_score": [
                            {"field_name": "add_1",
                            },
                            {"field_name": "add_2",
                            }
                        ]
                    })
                )
            ]
        )

        modifier_res = tensor_search.bulk_search(marqo_config=self.config, query=bulk_search_query_with_modifier)
        modifier_score = modifier_res["result"][0]["hits"][0]["_score"]
        self.assertEqual(normal_score, modifier_score)

    def get_expected_score(self, doc, ori_score, score_modifiers: ScoreModifier):
        add = 0.0
        if score_modifiers.multiply_score_by is not None:
            for config in score_modifiers.multiply_score_by:
                if config.field_name in doc:
                    if isinstance(doc[config.field_name], (int, float)):
                        ori_score = ori_score * config.weight * doc[config.field_name]

        if score_modifiers.add_to_score is not None:
            for config in score_modifiers.add_to_score:
                if config.field_name in doc:
                    if isinstance(doc[config.field_name], (int, float)):
                        add = add + config.weight * doc[config.field_name]

        return max(0.0, (ori_score + add))
    
    def test_bulk_search_score_modified_as_expected(self):
        documents = self.test_score_documents

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                    "filter"], auto_refresh=True)

        bulk_search_query = BulkSearchQuery(
            queries=[
                BulkSearchQueryEntity(
                    index=self.index_name,
                    q="what is the rider doing?",
                    limit=10
                )
            ]
        )

        normal_res = tensor_search.bulk_search(marqo_config=self.config, query=bulk_search_query)
        normal_score = normal_res["result"][0]["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            bulk_search_query_with_modifier = BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name,
                        q="what is the rider doing?",
                        limit=10,
                        scoreModifiers=score_modifiers
                    )
                ]
            )
            modifier_res = tensor_search.bulk_search(marqo_config=self.config, query=bulk_search_query_with_modifier)

            assert len(modifier_res["result"][0]["hits"]) == len(documents)
            for result in modifier_res["result"][0]["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_bulk_search_score_modified_as_expected_with_filter(self):
        documents = self.test_score_documents

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                    "filter"], auto_refresh=True)

        bulk_search_query = BulkSearchQuery(
            queries=[
                BulkSearchQueryEntity(
                    index=self.index_name,
                    q="what is the rider doing?",
                    limit=10
                )
            ]
        )

        normal_res = tensor_search.bulk_search(marqo_config=self.config, query=bulk_search_query)
        normal_score = normal_res["result"][0]["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            bulk_search_query_with_modifier = BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name,
                        q="what is the rider doing?",
                        limit=10,
                        scoreModifiers=score_modifiers,
                        filter="filter:original"
                    )
                ]
            )
            modifier_res = tensor_search.bulk_search(marqo_config=self.config, query=bulk_search_query_with_modifier)

            assert len(modifier_res["result"][0]["hits"]) == 1
            assert modifier_res["result"][0]["hits"][0]["_id"] == "0"
            for result in modifier_res["result"][0]["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError
    
    def test_bulk_search_score_modified_as_expected_with_searchable_attributes(self):
        documents = self.test_score_documents

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10,
                                          searchable_attributes=["my_image_field"])
        normal_res = tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(
            queries=[
                BulkSearchQueryEntity(
                    index=self.index_name,
                    q="what is the rider doing?",
                    limit=10,
                    searchableAttributes=["my_image_field"],
                    scoreModifiers=None,
                )
            ]
        )
        )["result"][0]
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name,
                        q="what is the rider doing?",
                        limit=10,
                        searchableAttributes=["my_image_field"],
                        scoreModifiers=score_modifiers,
                    )
                ]
            ))["result"][0]

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_bulk_search_score_modified_as_expected_with_attributes_to_retrieve(self):
        documents = self.test_score_documents
        invalid_fields = copy.deepcopy(documents[0])
        del invalid_fields["_id"]

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        normal_res = tensor_search.search(config=self.config, index_name=self.index_name,
                                          text="what is the rider doing?", score_modifiers=None, result_count=10,)
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:

            modifier_res = tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name,
                        q="what is the rider doing?",
                        limit=10,
                        attributesToRetrieve=["_id"],
                        scoreModifiers=score_modifiers,
                    )
                ]
            ))["result"][0]

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                for field in list(invalid_fields):
                    assert field not in result
                # need to get the original doc with score modifier fields to compute expected score
                original_doc = [doc for doc in documents if doc["_id"] == result["_id"]][0]
                expected_score = self.get_expected_score(original_doc, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_bulk_search_score_modified_as_expected_with_skipped_fields(self):
        # multiply_1 is string here, which should be skipped in modification
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 4 fields
             "multiply_1": "1",
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 30.0,
             "_id": "1"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # let's multiply by 0
             "multiply_1": "0",
             "multiply_2": 20.0,
             "add_1": 1.0,
             "add_2": 3.0,
             "_id": "2"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # 2 fields
             "multiply_2": 20.3,
             "add_1": 1.2,
             "_id": "3"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # change order
             "add_1": 1.0,
             "add_2": 3.0,
             "multiply_1": "1",
             "multiply_2": -20.0,
             "_id": "4"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             # float
             "multiply_1": "1",
             "multiply_2": 2.531,
             "add_1": 1.454,
             "add_2": -3.692,
             "_id": "5"
             },
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]

        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)

        normal_res = tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(
            queries=[
                BulkSearchQueryEntity(
                    index=self.index_name,
                    q="what is the rider doing?",
                    limit=10,
                    scoreModifiers=None,
                )
            ]
        ))["result"][0]
        normal_score = normal_res["hits"][0]["_score"]

        epsilon = 1e-5
        score_modifiers_list = self.test_valid_score_modifiers_list
        for score_modifiers in score_modifiers_list:
            modifier_res = tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name,
                        q="what is the rider doing?",
                        limit=10,
                        scoreModifiers=score_modifiers,
                    )
                ]
            ))["result"][0]

            assert len(modifier_res["hits"]) == len(documents)
            for result in modifier_res["hits"]:
                expected_score = self.get_expected_score(result, normal_score, score_modifiers)
                if abs(expected_score - result["_score"]) > epsilon:
                    raise AssertionError

    def test_bulk_search_normal_query_body_is_called(self):
        documents = [
            {"my_text_field": "A rider is riding a horse jumping over the barrier.",
             "my_image_field": "https://marqo-assets.s3.amazonaws.com/tests/images/image2.jpg",
             "_id": "0",
             "filter": "original"
             },
        ]
        add_docs_caller(config=self.config, index_name=self.index_name, docs=documents,
                                    non_tensor_fields=["multiply_1", "multiply_2", "add_1", "add_2",
                                                       "filter"], auto_refresh=True)
        def pass_create_normal_tensor_search_query(*arg, **kwargs):
            return _create_normal_tensor_search_query(*arg, **kwargs)

        mock_create_normal_tensor_search_query = mock.MagicMock()
        mock_create_normal_tensor_search_query.side_effect = pass_create_normal_tensor_search_query

        @mock.patch("marqo.tensor_search.tensor_search._create_normal_tensor_search_query", mock_create_normal_tensor_search_query)
        def run():
            tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name,
                        q="what is the rider doing?",
                        limit=10,
                        scoreModifiers=None,
                    )
                ]
            ))["result"][0]
            mock_create_normal_tensor_search_query.assert_called()

            return True
        assert run()

    def test_score_modifier_vector_text_search_verbose(self):
        mock_vector_text_search_verbose = mock.MagicMock()
        mock_vector_text_search_verbose.side_effect = _vector_text_search_query_verbose

        mock_pprint = mock.MagicMock()
        mock_pprint.side_effect = pprint.pprint

        mock_generate_verbose_one_body = mock.MagicMock()
        mock_generate_verbose_one_body.side_effect = _generate_vector_text_search_query_for_verbose_one
        @mock.patch('marqo.tensor_search.tensor_search._generate_vector_text_search_query_for_verbose_one', mock_generate_verbose_one_body)
        @mock.patch('marqo.tensor_search.tensor_search._vector_text_search_query_verbose', mock_vector_text_search_verbose)
        @mock.patch('marqo.tensor_search.tensor_search.pprint.pprint', mock_pprint)
        def run():
            for verbose in [0, 1, 2]:
                search_res = tensor_search.search(config=self.config, text="random text",
                                                  index_name=self.index_name, verbose=verbose,
                                                  score_modifiers=ScoreModifier(**{
                                                      "multiply_score_by":
                                                          [{"field_name": "multiply_1",
                                                            "weight": 1, },
                                                           {"field_name": "multiply_2", }
                                                           ],
                                                      "add_to_score": [
                                                          {"field_name": "add_1",
                                                           },
                                                          {"field_name": "add_2",
                                                           }
                                                      ]
                                                  }))

                if verbose == 0:
                    mock_vector_text_search_verbose.assert_not_called()
                elif verbose > 0:
                    mock_vector_text_search_verbose.assert_called_once()
                    _, verbose_kwargs = mock_vector_text_search_verbose.call_args
                    assert verbose_kwargs["verbose"] == verbose
                    assert verbose_kwargs["body"][0]["index"] == self.index_name
                    assert "knn" in \
                           verbose_kwargs["body"][1]["query"]["function_score"]["query"]["nested"]["query"]["function_score"][
                               "query"]

                    assert mock_pprint.call_count == 2

                    pprint_args, pprint_kwargs = mock_pprint.call_args_list[0]
                    if verbose == 1:
                        mock_generate_verbose_one_body.assert_called_once_with(verbose_kwargs["body"])
                    elif verbose == 2:
                        assert verbose_kwargs["body"] == pprint_args[0]
                        assert pprint_kwargs["compact"] == True

                mock_vector_text_search_verbose.reset_mock()
                mock_pprint.reset_mock()
            return True
        assert run()

    def test_score_modifier_with_changing_weights(self):
        """To test there is no compiling error when changing weights of score modifier and changing field names,
        but fixed number of fields."""
        try:
            tensor_search.delete_index(self.config, self.index_name)
        except IndexNotFoundError:
            pass

        index_settings = {
            "index_defaults": {
                # use a small model to speed up the test
                "model": "random/small"
            }
        }
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name, index_settings=index_settings)

        # By default, Opensearch only allow 75 recompiling in 5 minutes.
        for _ in range(300):
            res = tensor_search.search(config=self.config, text="random text", score_modifiers=ScoreModifier(**{
                "multiply_score_by": [{"field_name": f"multiply_{np.random.randint(1, 100)}", "weight": np.random.rand()},
                                      {"field_name": f"multiply_{np.random.randint(1, 100)}", "weight": np.random.rand()}],
                "add_to_score": [{"field_name": f"add_{np.random.randint(1, 100)}", "weight": np.random.rand()},
                                 {"field_name": f"add_{np.random.randint(1, 100)}", "weight": np.random.rand()}]
            }), index_name=self.index_name)

    def test_too_many_dynamic_script_compilation_error(self):
        """To test the dynamic script compilation errors in Opensearch.
        The error is expected to be raised when there are too many dynamic script compilation in a short time."""
        try:
            tensor_search.delete_index(self.config, self.index_name)
        except IndexNotFoundError:
            pass

        index_settings = {
            "index_defaults": {
                # use a small model to speed up the test
                "model": "random/small"
            }
        }
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name, index_settings=index_settings)

        # By default, Opensearch only allow 75 recompiling in 5 minutes.
        multiply_score_by_list = []
        for i in range(300):
            multiply_score_by_list.append({"field_name": f"multiply_{i}", "weight": np.random.rand()})

            try:
                res = tensor_search.search(config=self.config, text="random text", score_modifiers=ScoreModifier(**{
                    "multiply_score_by": multiply_score_by_list,
                }), index_name=self.index_name)
                if i >= 250:
                    assert False, "Should raise BackendCommunicationError"
            except BackendCommunicationError as e:
                assert "Too many dynamic script compilations" in str(e.message)
                break

    def test_score_modifiers_script(self):
        """Ensure that the script generated by score modifiers is correct."""
        mock_generate_script_score = mock.MagicMock()

        generated_script_score = []
        def record_script_score_output(*args, **kwargs):
            generated_script_score.append(_create_score_modifiers_tensor_search_query(*args, **kwargs))
            return generated_script_score[-1]
        mock_generate_script_score.side_effect = record_script_score_output
        @mock.patch("marqo.tensor_search.tensor_search._create_score_modifiers_tensor_search_query", mock_generate_script_score)
        def run():
            first_call = tensor_search.search(config=self.config, text="random text", score_modifiers=ScoreModifier(**{
                "multiply_score_by": [{"field_name": f"multiply_{np.random.randint(1, 100)}", "weight": np.random.rand()},
                                      {"field_name": f"multiply_{np.random.randint(1, 100)}", "weight": np.random.rand()}],
                "add_to_score": [{"field_name": f"add_{np.random.randint(1, 100)}", "weight": np.random.rand()},
                                 {"field_name": f"add_{np.random.randint(1, 100)}", "weight": np.random.rand()}]
            }), index_name=self.index_name)

            second_call = tensor_search.search(config=self.config, text="random text", score_modifiers=ScoreModifier(**{
                "multiply_score_by": [{"field_name": f"multiply_{np.random.randint(1, 100)}", "weight": np.random.rand()},
                                      {"field_name": f"multiply_{np.random.randint(1, 100)}", "weight": np.random.rand()}],
                "add_to_score": [{"field_name": f"add_{np.random.randint(1, 100)}", "weight": np.random.rand()},
                                 {"field_name": f"add_{np.random.randint(1, 100)}", "weight": np.random.rand()}]
            }), index_name=self.index_name)

            assert generated_script_score[0]["query"]["function_score"]["query"]["nested"]["query"]["function_score"]["functions"][0]["script_score"]["script"]["source"] == \
            generated_script_score[1]["query"]["function_score"]["query"]["nested"]["query"]["function_score"]["functions"][0]["script_score"]["script"]["source"]
            return True

        assert run()

    def test_multiply_score_by_operator(self):
        op = ScoreModifierOperator(field_name="test", weight=2.0)
        script, weights, fields = op.to_painless_script_and_params(1, "multiply_score_by")

        self.assertIn("multiplier_field_1", script)
        self.assertEqual(weights, {"multiplier_weight_1": 2.0})
        self.assertEqual(fields, {"multiplier_field_1": "__chunks.test"})

    def test_add_to_score_operator(self):
        op = ScoreModifierOperator(field_name="test", weight=2.0)
        script, weights, fields = op.to_painless_script_and_params(1, "add_to_score")

        self.assertIn("add_field_1", script)
        self.assertEqual(weights, {"add_weight_1": 2.0})
        self.assertEqual(fields, {"add_field_1": "__chunks.test"})

    def test_invalid_operation(self):
        op = ScoreModifierOperator(field_name="test", weight=2.0)

        with self.assertRaises(ValueError) as context:
            op.to_painless_script_and_params(1, "invalid_operation")
        self.assertIn("operation must be either 'multiply_score_by' or 'add_to_score'", str(context.exception))

    def test_field_name_validation(self):
        with self.assertRaises(InvalidArgError) as context:
            ScoreModifierOperator(field_name="_id", weight=2.0)
        self.assertIn("_id is not allowed as a field_name", str(context.exception))

    def test_multiply_score_by_script_score(self):
        modifier = ScoreModifier(multiply_score_by=[ScoreModifierOperator(field_name="test1", weight=2.0)])
        result = modifier.to_script_score()

        self.assertIn("multiplier_field_0", result["source"])
        self.assertIn("multiplier_weight_0", result["params"])
        self.assertEqual(result["params"]["multiplier_weight_0"], 2.0)
        self.assertEqual(result["params"]["multiplier_field_0"], "__chunks.test1")

    def test_add_to_score_script_score(self):
        modifier = ScoreModifier(add_to_score=[ScoreModifierOperator(field_name="test2", weight=3.0)])
        result = modifier.to_script_score()

        self.assertIn("add_field_0", result["source"])
        self.assertIn("add_weight_0", result["params"])
        self.assertEqual(result["params"]["add_weight_0"], 3.0)
        self.assertEqual(result["params"]["add_field_0"], "__chunks.test2")

    def test_both_operations_script_score(self):
        modifier = ScoreModifier(
            multiply_score_by=[ScoreModifierOperator(field_name="test1", weight=2.0)],
            add_to_score=[ScoreModifierOperator(field_name="test2", weight=3.0)]
        )
        result = modifier.to_script_score()

        self.assertIn("multiplier_field_0", result["source"])
        self.assertIn("add_field_0", result["source"])
        self.assertIn("multiplier_weight_0", result["params"])
        self.assertIn("add_weight_0", result["params"])
        self.assertEqual(result["params"]["multiplier_weight_0"], 2.0)
        self.assertEqual(result["params"]["add_weight_0"], 3.0)
        self.assertEqual(result["params"]["multiplier_field_0"], "__chunks.test1")
        self.assertEqual(result["params"]["add_field_0"], "__chunks.test2")

    def test_multiple_multiply_score_by_script_score(self):
        modifier = ScoreModifier(
            multiply_score_by=[
                ScoreModifierOperator(field_name="test1", weight=2.0),
                ScoreModifierOperator(field_name="test2", weight=3.0)
            ]
        )
        result = modifier.to_script_score()

        self.assertIn("multiplier_field_0", result["source"])
        self.assertIn("multiplier_field_1", result["source"])
        self.assertEqual(result["params"]["multiplier_weight_0"], 2.0)
        self.assertEqual(result["params"]["multiplier_weight_1"], 3.0)

    def test_multiple_add_to_score_script_score(self):
        modifier = ScoreModifier(
            add_to_score=[
                ScoreModifierOperator(field_name="test1", weight=2.0),
                ScoreModifierOperator(field_name="test2", weight=3.0)
            ]
        )
        result = modifier.to_script_score()

        self.assertIn("add_field_0", result["source"])
        self.assertIn("add_field_1", result["source"])
        self.assertEqual(result["params"]["add_weight_0"], 2.0)
        self.assertEqual(result["params"]["add_weight_1"], 3.0)

    def test_zero_weight_multiply_score_by_script_score(self):
        modifier = ScoreModifier(multiply_score_by=[ScoreModifierOperator(field_name="test1", weight=0.0)])
        result = modifier.to_script_score()
        self.assertEqual(result["params"]["multiplier_weight_0"], 0.0)

    def test_negative_weight_add_to_score_script_score(self):
        modifier = ScoreModifier(add_to_score=[ScoreModifierOperator(field_name="test2", weight=-3.0)])
        result = modifier.to_script_score()
        self.assertEqual(result["params"]["add_weight_0"], -3.0)