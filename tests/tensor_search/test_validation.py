import os
import unittest
from enum import Enum
from unittest import mock
from unittest.mock import patch

from marqo.api.exceptions import (
    InvalidFieldNameError, InvalidDocumentIdError, InvalidArgError, DocTooLargeError
)
from marqo import exceptions as base_exceptions
from marqo.tensor_search import enums
from marqo.tensor_search import validation
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest
from marqo.tensor_search.models.score_modifiers_object import ScoreModifier
from marqo.tensor_search.models.search import SearchContext
from pydantic import ValidationError


class TestValidation(unittest.TestCase):

    def setUp(self) -> None:
        class SimpleEnum(str, Enum):
            ABC = "APPLE"
            DEF = "BANANA"

        self.SimpleEnum = SimpleEnum

    def test_validate_str_against_enum_case_senstive(self):
        try:
            validation.validate_str_against_enum("banana", self.SimpleEnum, case_sensitive=True)
            raise AssertionError
        except ValueError:
            pass

    def test_validate_str_against_enum_case_insensitive(self):
        assert "banana" == validation.validate_str_against_enum("banana", self.SimpleEnum, case_sensitive=False)

    def test_validate_str_against_enum(self):
        assert "APPLE" == validation.validate_str_against_enum("APPLE", self.SimpleEnum)

    def test_validate_chunk_plus_name(self):
        try:
            validation.validate_field_name("__chunks.__field_name")
            raise AssertionError
        except InvalidFieldNameError as s:
            pass

    def test_nesting_attempt(self):
        try:
            validation.validate_field_name("some_object.__field_name")
            raise AssertionError
        except InvalidFieldNameError as s:
            pass

    def test_validate_field_name_good(self):
        assert "some random fieldname" == validation.validate_field_name("some random fieldname")

    def test_validate_field_name_good_2(self):
        assert "abc__field_name" == validation.validate_field_name("abc__field_name")

    def test_validate_field_name_empty(self):
        try:
            validation.validate_field_name("")
            raise AssertionError
        except InvalidFieldNameError as s:
            pass

    def test_validate_field_name_none(self):
        try:
            validation.validate_field_name(None)
            raise AssertionError
        except InvalidFieldNameError as s:
            pass

    def test_validate_field_name_other(self):
        try:
            validation.validate_field_name(123)
            raise AssertionError
        except InvalidFieldNameError as s:
            assert "must be str" in str(s)

    def test_validate_field_name_protected(self):
        try:
            validation.validate_field_name("__field_name")
            raise AssertionError
        except InvalidFieldNameError as s:
            assert "protected field" in str(s)

    def test_validate_field_name_vector_prefix(self):
        try:
            validation.validate_field_name("__vector_")
            raise AssertionError
        except InvalidFieldNameError as s:
            assert "protected prefix" in str(s)

    def test_validate_field_name_vector_prefix_2(self):
        try:
            validation.validate_field_name("__vector_abc")
            raise AssertionError
        except InvalidFieldNameError as s:
            assert "protected prefix" in str(s)

    def test_validate_doc_empty(self):
        try:
            validation.validate_doc({})
            raise AssertionError
        except InvalidArgError as s:
            pass

    def test_validate_field_name_highlight(self):
        bad_name = "_highlights"
        try:
            validation.validate_field_name(bad_name)
            raise AssertionError
        except InvalidFieldNameError as s:
            assert 'protected field' in str(s)

    def test_validate_field_content_bad(self):
        bad_field_content = [
            {123}, None, ['not 100% strings', 134, 1.4, False],
            ['not 100% strings', True]
        ]
        for non_tensor_field in (True, False):
            for bad_content in bad_field_content:
                try:
                    validation.validate_field_content(bad_content, is_non_tensor_field=non_tensor_field)
                    raise AssertionError
                except InvalidArgError as e:
                    pass

    def test_validate_field_content_good(self):
        good_field_content = [
            123, "heehee", 12.4, False
        ]
        for non_tensor_field in (True, False):
            for good_content in good_field_content:
                assert good_content == validation.validate_field_content(good_content,
                                                                         is_non_tensor_field=non_tensor_field)

    def test_validate_field_content_list(self):
        good_field_content = [
            [], [''], ['abc', 'efg', '123'], ['', '']
        ]
        for good_content in good_field_content:
            assert good_content == validation.validate_field_content(good_content, is_non_tensor_field=True)

        for good_content in good_field_content:
            # fails when non tensor field
            try:
                validation.validate_field_content(good_content, is_non_tensor_field=False)
                raise AssertionError
            except InvalidArgError:
                pass

    def test_validate_id_good(self):
        bad_ids = [
            {123}, [], None, {"abw": "cjnk"}, 1234
        ]
        for bad_content in bad_ids:
            try:
                validation.validate_id(bad_content)
                raise AssertionError
            except InvalidDocumentIdError as e:
                pass

    def test_validate_id_bad(self):
        good_ids = [
            "123", "hehee", "12_349"
        ]
        for good_content in good_ids:
            assert good_content == validation.validate_id(good_content)

    def test_validate_doc_max_size(self):
        max_size = 1234567
        mock_environ = {enums.EnvVars.MARQO_MAX_DOC_BYTES: str(max_size)}

        @mock.patch.dict(os.environ, mock_environ)
        def run():
            good_doc = {"abcd": "a" * (max_size - 500)}
            good_back = validation.validate_doc(doc=good_doc)
            assert good_back == good_doc

            bad_doc = {"abcd": "a" * max_size}
            try:
                validation.validate_doc(doc=bad_doc)
                raise AssertionError
            except DocTooLargeError:
                pass
            return True

        assert run()

    def test_boost_validation_illegal(self):
        bad_boosts = [
            set(), (), {'': [1.2]},
            {'fine': [1.2], "ok": [1.2, -3], 'bad': [3, 1, -4]},
            {'fine': [1.2], "ok": [1.2, -3], 'bad': []},
            {'fine': [1.2], "ok": [1.2, -3], 'bad': ['1iu']},
            {'bad': ['str']}, {'bad': []}, {'bad': [1, 4, 5]},
        ]
        for search_method in ('TENSOR', 'LEXICAL', 'OTHER'):
            for bad_boost in bad_boosts:
                try:
                    validation.validate_boost(boost=bad_boost, search_method=search_method)
                    raise AssertionError
                except (InvalidArgError, InvalidFieldNameError) as e:
                    pass

    def test_boost_validation_good_boost_bad_method(self):
        good_boosts = [
            {}, {'fine': [1.2], "ok": [1.2, -3]}, {'fine': [1.2]}, {'fine': [1.2, -1]},
            {'fine': [0, 0]}, {'fine': [0]}, {'fine': [-1.3]}
        ]
        for search_method in ('', 'LEXICAL', 'OTHER'):
            for good_boost in good_boosts:
                try:

                    validation.validate_boost(boost=good_boost, search_method=search_method)
                    raise AssertionError
                except (InvalidArgError, InvalidFieldNameError) as e:
                    pass

    def test_boost_validation_good_boosts(self):
        good_boosts = [
            {}, {'fine': [1.2], "ok": [1.2, -3]}, None, {'fine': [1.2]}, {'fine': [1.2, -1]},
        ]
        for good_boost in good_boosts:
            assert good_boost == validation.validate_boost(boost=good_boost, search_method='TENSOR')

    def test_boost_validation_None_ok(self):
        for search_method in ('', 'LEXICAL', 'OTHER', 'TENSOR'):
            assert None is validation.validate_boost(boost=None, search_method=search_method)


class TestValidateSearchableAttributes(unittest.TestCase):

    def setUp(self) -> None:
        self.searchable_attributes = [f"field{i}" for i in range(5)]

    def test_search_method_not_tensor(self):
        validation.validate_searchable_attributes(
            self.searchable_attributes,
            search_method=enums.SearchMethod.LEXICAL
        )

    def test_maximum_searchable_attributes_not_set(self):
        validation.validate_searchable_attributes(
            self.searchable_attributes,
            search_method=enums.SearchMethod.TENSOR
        )

    @patch.dict('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': '1'}})
    def test_searchable_attributes_is_none_max_value_set_raise_invalid_arg_error(self):
        try:
            validation.validate_searchable_attributes(
                searchable_attributes=None,
                search_method=enums.SearchMethod.TENSOR
            )
            raise AssertionError("'searchable_attributes' is None, but MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES is set")

        except InvalidArgError as e:
            self.assertTrue("MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES" in e.message)

    @patch.dict('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': '1'}})
    def test_searchable_attributes_not_set_but_max_attributes_set__raise_(self):
        with self.assertRaises(InvalidArgError):
            validation.validate_searchable_attributes(
                searchable_attributes=None,
                search_method=enums.SearchMethod.TENSOR
            )

    @patch.dict('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': '1'}})
    def test_searchable_attributes_set__use_searchable_attributes(self):
        with self.assertRaises(InvalidArgError):
            validation.validate_searchable_attributes(
                searchable_attributes=self.searchable_attributes,
                search_method=enums.SearchMethod.TENSOR
            )

    @patch.dict('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': '6'}})
    def test_searchable_attributes_below_limit(self):
        validation.validate_searchable_attributes(
            searchable_attributes=self.searchable_attributes,
            search_method=enums.SearchMethod.TENSOR
        )


class TestValidateIndexSettings(unittest.TestCase):

    @staticmethod
    def get_good_index_settings():
        return {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 2,
                    "split_overlap": 0,
                    "split_method": "sentence"
                },
                "image_preprocessing": {
                    "patch_method": None
                }
            },
            "number_of_shards": 5,
            "number_of_replicas": 1
        }

    def test_validate_mappings(self):
        mappings = [
            {
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": 0.5
                    }
                }
            },
            {
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": 0.5
                    }
                },
                "other_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": 0.7,
                        "bugs": 200
                    }
                },
            },
            {},
            {
                " ": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": -2
                    }
                }
            },
            {
                "abcd ": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": -4.6,
                        "other_text": 22
                    }
                }
            },
            {
                "abcd ": {
                    "type": "multimodal_combination",
                    "weights": {}
                }
            },
            {
                "abcd ": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": 0,
                    }
                }
            },

            # Mappings with custom vector
            {
                "my_custom_vector": {
                    "type": "custom_vector"
                }
            },
            # Mappings with both custom vector and multimodal combination
            {
                "my_custom_vector": {
                    "type": "custom_vector"
                },
                "abcd ": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": -4.6,
                        "other_text": 22
                    }
                },
                "my_custom_vector_2": {
                    "type": "custom_vector"
                }
            },
        ]
        for d in mappings:
            assert d == validation.validate_mappings_object(d)

    def test_validate_mappings_invalid(self):
        mappings = [
            {
                "my_combination_field": {
                    "type": "othertype",  # bad type
                    "weights": {
                        "some_text": 0.5

                    }
                }
            },
            # Field with no type
            {
                "my_combination_field": {
                    "weights": {
                        "some_text": 0.5

                    }
                }
            },
            # Empty mapping
            {
                "empty field": {}
            },
            {
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "non_weights": {  # unknown fieldname 'non_weights' config in multimodal_combination
                        "some_text": 0.5
                    }
                }
            },
            {
                "my_combination_field": {
                    "type": "multimodal_combination",
                    # missing weights for multimodal_combination
                }
            },
            {
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {"blah": "woo"}  # non-number weights
                }
            },
            {
                "my_combination_field": {
                    "type": "multimodal_combination",
                    "weights": {"blah": "1.3"}  # non-number weights
                }
            },
            {
                "abcd ": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": -4.6,
                        "other_text": 22
                    },
                    "extra_field": {"blah"}  # unknown field
                }
            },
            {
                "abcd ": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": -4.6,
                        "other_text": 22,
                        "nontext": True  # non-number
                    },
                }
            },
            {  # needs more nesting
                "type": "multimodal_combination",
                "weights": {
                    "some_text": 0.5
                }
            },
            {
                "my_combination_field": {  # this dict is OK
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": 0.5
                    }
                },
                "other_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": 0.7,
                        "bugs": [0.5, -1.3]  # this is bad array
                    }
                },
            },
            # Custom vector with extra field
            {
                "my_custom_vector": {
                    "type": "custom_vector",
                    "extra_field": "blah"
                }
            },
            # Custom vector with extra field and multimodal
            {
                "my_custom_vector": {
                    "type": "custom_vector",
                    "extra_field_2": "blah"
                },
                "abcd": {
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": -4.6,
                        "other_text": 22
                    }
                }
            },
        ]
        for mapping in mappings:
            try:
                validation.validate_mappings_object(mapping)
                raise AssertionError
            except InvalidArgError as e:
                pass

    def test_validate_multimodal_combination_mappings_object(self):
        mappings = [
            {
                "type": "multimodal_combination",
                "weights": {
                    "some_text": 0.5
                }
            },
            {
                "type": "multimodal_combination",
                "weights": {
                    "some_text": -2
                }
            },
            {
                "type": "multimodal_combination",
                "weights": {
                    "some_text": -4.6,
                    "other_text": 22
                }
            },
            {
                "type": "multimodal_combination",
                "weights": {}
            },
            {
                "type": "multimodal_combination",
                "weights": {
                    "some_text": 0,
                }
            },
        ]
        for d in mappings:
            assert d == validation.validate_multimodal_combination_mappings_object(d)

    def test_invalid_multimodal_combination_mappings_object(self):
        mappings = [
            ({
                "my_combination_field": { # valid mappings dir, but not valid multimodal
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": 0.5
                    }
                }
            }, "'type' is a required property"),
            ({
                "type": "othertype",  # bad type
                "weights": {
                    "some_text": 0.5

                }
            }, "'othertype' is not one of"),
            ({
                "type": "multimodal_combination",
                "non_weights": {  # unknown fieldname 'non_weights' config in multimodal_combination
                    "some_text": 0.5
                }
            }, "'weights' is a required property"),
            ({
                "type": "multimodal_combination",
                # missing weights for multimodal_combination
            }, "'weights' is a required property"),
            ({
                "type": "multimodal_combination",
                "weights": {"blah": "woo"}  # non-number weights
            }, "is not of type 'number'"),
            ({
                "type": "multimodal_combination",
                "weights": {"blah": "1.3"}  # non-number weights
            }, "is not of type 'number'"),
            ({
                "type": "multimodal_combination",
                "weights": {
                    "some_text": -4.6,
                    "other_text": 22
                },
                "extra_field": {"blah"}  # unknown field
            }, "Additional properties are not allowed"),
            ({
                "type": "multimodal_combination",
                "weights": {
                    "some_text": -4.6,
                    "other_text": 22,
                    "nontext": True  # non-number
                },
            }, "is not of type 'number'")
        ]
        for mapping, error_message in mappings:
            try:
                validation.validate_multimodal_combination_mappings_object(mapping)
                raise AssertionError
            except InvalidArgError as e:
                assert error_message in e.message

    def test_valid_custom_vector_mappings_object(self):
        # There is only 1 valid format for custom vector mapping.
        mappings = [
            {
                "type": "custom_vector"
            }
        ]
        for d in mappings:
            assert d == validation.validate_custom_vector_mappings_object(d)

    def test_invalid_custom_vector_mappings_object(self):
        mappings = [
            # Extra field
            ({
                 "type": "custom_vector",
                 "extra_field": "blah"
             }, "Additional properties are not allowed ('extra_field' was unexpected)"),
            # Misspelled type field
            ({
                 "typeblahblah": "custom_vector",
             }, "'type' is a required property"),
            # Type not custom_vector
            ({
                 "type": "the wrong field type",
             }, "'the wrong field type' is not one of"),
            # Empty
            ({}, "'type' is a required property")
        ]
        for mapping, error_message in mappings:
            try:
                validation.validate_custom_vector_mappings_object(mapping)
                raise AssertionError
            except InvalidArgError as e:
                assert error_message in e.message

    def test_validate_valid_context_object(self):
        valid_context_list = [
            {
                "tensor": [
                    {"vector": [0.2132] * 512, "weight": 0.32},
                    {"vector": [0.2132] * 512, "weight": 0.32},
                    {"vector": [0.2132] * 512, "weight": 0.32},
                ]
            },
            {
                "tensor": [
                    {"vector": [0.2132] * 512, "weight": 1},
                    {"vector": [0.2132] * 512, "weight": 1},
                    {"vector": [0.2132] * 512, "weight": 1},
                ]
            },

            {
                # Note we are not validating the vector size here
                "tensor": [
                    {"vector": [0.2132] * 53, "weight": 1},
                    {"vector": [23, ], "weight": 1},
                    {"vector": [0.2132] * 512, "weight": 1},
                ],
                "addition_field": None
            },
            {
                "tensor": [
                    {"vector": [0.2132] * 53, "weight": 1},
                    {"vector": [23, ], "weight": 1},
                    {"vector": [0.2132] * 512, "weight": 1},
                ],
                "addition_field_1": None,
                "addition_field_2": "random"
            },
            {
                "tensor": [
                              {"vector": [0.2132] * 512, "weight": 0.32},
                          ] * 64
            },
        ]

        for valid_context in valid_context_list:
            SearchContext(**valid_context)

    def test_validate_invalid_context_object(self):
        valid_context_list = [
            # {
            #     # Typo in tensor
            #     "tensors": [
            #         {"vector" : [0.2132] * 512, "weight" : 0.32},
            #         {"vector": [0.2132] * 512, "weight": 0.32},
            #         {"vector": [0.2132] * 512, "weight": 0.32},
            #     ]
            # },
            {
                # Typo in vector
                "tensor": [
                    {"vectors": [0.2132] * 512, "weight": 1},
                    {"vector": [0.2132] * 512, "weight": 1},
                    {"vector": [0.2132] * 512, "weight": 1},
                ]
            },
            {
                # Typo in weight
                "tensor": [
                    {"vector": [0.2132] * 53, "weight": 1},
                    {"vector": [23, ], "weight": 1},
                    {"vector": [0.2132] * 512, "weights": 1},
                ],
                "addition_field": None
            },
            {
                # Int instead of list
                "tensor": [
                    {"vector": [0.2132] * 53, "weight": 1},
                    {"vector": [23, ], "weight": 1},
                    {"vector": 3, "weight": 1},
                ],
                "addition_field_1": None,
                "addition_field_2": "random"
            },
            {
                # Str instead of list
                "tensor": [
                    {"vector": str([0.2132] * 512), "weight": 0.32},
                    {"vector": [0.2132] * 512, "weight": 0.32},
                    {"vector": [0.2132] * 512, "weight": 0.32},
                ],
                "addition_field_1": None,
                "addition_field_2": "random"
            },
            {
                # None instead of list
                "tensor": [
                    {"vector": [0.2132] * 53, "weight": 1},
                    {"vector": [23, ], "weight": 1},
                    {"vectors": None, "weight": 1},
                ],
                "addition_field_1": None,
                "addition_field_2": "random"
            },
            {
                # too many vectors, maximum 64
                "tensor": [
                              {"vector": [0.2132] * 512, "weight": 0.32},
                          ] * 65
            },
            {
                # None
                "tensor": None,
            },
            {
                # Empty tensor
                "tensor": [],
            },
        ]

        for invalid_context in valid_context_list:
            try:
                s = SearchContext(**invalid_context)
                raise AssertionError(invalid_context, s)
            except InvalidArgError:
                pass

    def test_invalid_custom_score_fields(self):
        invalid_custom_score_fields_list = [
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
                     }, ],
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
            {},  # empty
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
        for invalid_custom_score_fields in invalid_custom_score_fields_list:
            try:
                v = ScoreModifier(**invalid_custom_score_fields)
                raise AssertionError(invalid_custom_score_fields, v)
            except InvalidArgError:
                pass

    def test_valid_custom_score_fields(self):
        valid_custom_score_fields_list = [
            {
                "multiply_score_by":
                    [{"field_name": "reputation",
                      "weight": 1,
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],

                "add_to_score": [
                    {"field_name": "rate",
                     }]
            },
            {
                "multiply_score_by":
                    [{"field_name": "reputation",
                      },
                     {
                         "field_name": "reputation-test",
                     }, ],

                "add_to_score": [
                    {"field_name": "rate",
                     }]
            },
            {
                # miss one part
                "add_to_score": [
                    {"field_name": "rate",
                     }]
            },
        ]

        for valid_custom_score_fields in valid_custom_score_fields_list:
            ScoreModifier(**valid_custom_score_fields)

    def test_validate_dict(self):
        """
        Only dict content type accepted is `custom_vector`. Other dict types may be added in the future.
        """
        test_mappings = {
            "my_custom_vector": {
                "type": "custom_vector"
            }
        }

        # ============== custom vector validate_dict tests ==============
        index_model_dimensions = 384
        # custom vector, valid
        obj = {"content": "custom content is here!!", "vector": [1.0 for _ in range(index_model_dimensions)]}
        assert validation.validate_dict(field="my_custom_vector",
                                        field_content=obj,
                                        is_non_tensor_field=False,
                                        mappings=test_mappings,
                                        index_model_dimensions=index_model_dimensions) == obj

        # custom vector, valid (no content). must be filled with empty string
        obj = {"vector": [1.0 for _ in range(index_model_dimensions)]}
        assert validation.validate_dict(field="my_custom_vector",
                                        field_content=obj,
                                        is_non_tensor_field=False,
                                        mappings=test_mappings,
                                        index_model_dimensions=index_model_dimensions) \
               == {"content": "", "vector": [1.0 for _ in range(index_model_dimensions)]}

        invalid_custom_vector_objects = [
            # Wrong vector length
            ({"content": "custom content is here!!", "vector": [1.0, 1.0, 1.0]}, "given vector is of length"),
            ({"content": "custom content is here!!", "vector": [1.0] * 1000}, "given vector is of length"),
            # Wrong content type
            ({"content": 12345, "vector": [1.0 for _ in range(index_model_dimensions)]},
             "must be one of the following types"),
            # Wrong vector type inside list (even if correct length)
            ({"content": "custom content is here!!",
              "vector": [1.0 for _ in range(index_model_dimensions - 1)] + ["NOT A FLOAT"]},
             "must be a list of numbers"),
            # Field that shouldn't be there
            ({"content": "custom content is here!!", "vector": [1.0 for _ in range(index_model_dimensions)],
              "extra_field": "blah"}, "unexpected extra fields"),
            # No vector
            ({"content": "custom content is here!!"}, "missing 'vector'"),
            # Nested dict inside custom vector content
            ({
                 "content": {
                     "content": "custom content is here!!",
                     "vector": [1.0 for _ in range(index_model_dimensions)]
                 },
                 "vector": [1.0 for _ in range(index_model_dimensions)]
             }, "must be one of the following types"),
        ]
        for case, error_message in invalid_custom_vector_objects:
            with self.subTest(f"case={case}, error_message={error_message}"):
                try:
                    validation.validate_dict(field="my_custom_vector",
                                             field_content=case,
                                             is_non_tensor_field=False,
                                             mappings=test_mappings,
                                             index_model_dimensions=index_model_dimensions)
                    raise AssertionError
                except ValidationError as e:
                    assert error_message in str(e.args[0][0].exc)

        # No index model dimensions
        with self.subTest("No index model dimensions"):
            with self.assertRaises(ValidationError) as cm:
                validation.validate_dict(field="my_custom_vector",
                                         field_content={"content": "custom content is here!!",
                                                        "vector": [1.0 for _ in range(index_model_dimensions)]},
                                         is_non_tensor_field=False,
                                         mappings=test_mappings,
                                         index_model_dimensions=None)
            self.assertIn("none is not an allowed value", str(cm.exception.args[0][0].exc))

        # Non-int index model dimensions
        with self.subTest("No index model dimensions"):
            with self.assertRaises(ValidationError) as cm:
                validation.validate_dict(field="my_custom_vector",
                                         field_content={"content": "custom content is here!!",
                                                        "vector": [1.0 for _ in range(index_model_dimensions)]},
                                         is_non_tensor_field=False,
                                         mappings=test_mappings,
                                         index_model_dimensions="wrong type")

            self.assertIn("value is not a valid integer", str(cm.exception.args[0][0].exc))


class TestValidateDeleteDocsRequest(unittest.TestCase):

    def setUp(self) -> None:
        self.max_delete_docs_count = 10

    def test_valid_delete_request(self):
        delete_request = MqDeleteDocsRequest(index_name="my_index",
                                             schema_name='my__00index', document_ids=["id1", "id2", "id3"])
        result = validation.validate_delete_docs_request(delete_request, self.max_delete_docs_count)
        self.assertEqual(delete_request, result)

    def test_invalid_delete_request_not_instance(self):
        delete_request = {"index_name": "my_index", "document_ids": ["id1", "id2", "id3"], "auto_refresh": True}
        with self.assertRaises(RuntimeError):
            validation.validate_delete_docs_request(delete_request, self.max_delete_docs_count)

    def test_invalid_max_delete_docs_count(self):
        delete_request = MqDeleteDocsRequest(index_name="my_index",
                                             schema_name='my__00index',
                                             document_ids=["id1", "id2", "id3"])
        with self.assertRaises(RuntimeError):
            validation.validate_delete_docs_request(delete_request, "10")

    def test_empty_document_ids(self):
        delete_request = MqDeleteDocsRequest(index_name="my_index", schema_name='my__00index', document_ids=[])
        with self.assertRaises(InvalidDocumentIdError):
            validation.validate_delete_docs_request(delete_request, self.max_delete_docs_count)

    def test_document_ids_not_sequence(self):
        delete_request = MqDeleteDocsRequest(index_name="my_index", schema_name='my__00index', document_ids="id1")
        with self.assertRaises(InvalidArgError):
            validation.validate_delete_docs_request(delete_request, self.max_delete_docs_count)

    def test_exceed_max_delete_docs_count(self):
        delete_request = MqDeleteDocsRequest(index_name="my_index", schema_name='my__00index',
                                             document_ids=["id{}".format(i) for i in range(1, 12)])
        with self.assertRaises(InvalidArgError):
            validation.validate_delete_docs_request(delete_request, self.max_delete_docs_count)

    def test_invalid_document_id_type(self):
        delete_request = MqDeleteDocsRequest(index_name="my_index", schema_name='my__00index',
                                             document_ids=["id1", 2, "id3"])
        with self.assertRaises(InvalidDocumentIdError):
            validation.validate_delete_docs_request(delete_request, self.max_delete_docs_count)

    def test_empty_document_id(self):
        delete_request = MqDeleteDocsRequest(index_name="my_index", schema_name='my__00index',
                                             document_ids=["id1", "", "id3"])
        with self.assertRaises(InvalidDocumentIdError):
            validation.validate_delete_docs_request(delete_request, self.max_delete_docs_count)

    def test_no_limit(self):
        # the default limit is 10000,
        delete_request = MqDeleteDocsRequest(
            index_name="my_index", schema_name='my__00index',
            document_ids=["id{}".format(i) for i in range(1, 20000)])
        with self.assertRaises(RuntimeError):
            validation.validate_delete_docs_request(delete_request, None)
