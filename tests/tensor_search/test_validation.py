from marqo.tensor_search import validation
from enum import Enum
from marqo.tensor_search import enums
import unittest
import copy
from unittest import mock
from marqo.errors import (
    MarqoError, InvalidFieldNameError, InternalError,
    InvalidDocumentIdError, InvalidArgError, DocTooLargeError,
    InvalidIndexNameError
)
import pprint


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

    def test_validate_vector_name(self):
        good_name = "__vector_Title 1"
        assert good_name == validation.validate_vector_name(good_name)

    def test_validate_vector_name_2(self):
        """should only try removing the first prefix"""
        good_name = "__vector___vector_1"
        assert good_name == validation.validate_vector_name(good_name)

    def test_validate_vector_name_only_prefix(self):
        bad_vec = "__vector_"
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except InternalError as s:
            assert "empty" in str(s)

    def test_validate_vector_empty(self):
        bad_vec = ""
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except InternalError as s:
            assert "empty" in str(s)

    def test_validate_vector_int(self):
        bad_vec = 123
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except InternalError as s:
            assert 'must be str' in str(s)

        bad_vec_2 = ["efg"]
        try:
            validation.validate_vector_name(bad_vec_2)
            raise AssertionError
        except InternalError as s:
            assert 'must be str' in str(s)

    def test_validate_vector_no_prefix(self):
        bad_vec = "some bad title"
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except InternalError as s:
            assert 'vectors must begin' in str(s)

    def test_validate_vector_name_protected_field(self):
        """the vector name without the prefix can't be the name of a protected field"""
        bad_vec = "__vector___chunk_ids"
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except InternalError as s:
            assert 'protected name' in str(s)

    def test_validate_vector_name_id_field(self):
        bad_vec = "__vector__id"
        try:
            validation.validate_vector_name(bad_vec)
            raise AssertionError
        except InternalError as s:
            assert 'protected name' in str(s)

    def test_validate_field_name_highlight(self):
        bad_name = "_highlights"
        try:
            validation.validate_field_name(bad_name)
            raise AssertionError
        except InvalidFieldNameError as s:
            assert 'protected field' in str(s)

    def test_validate_field_content_bad(self):
        bad_field_content = [
            {123}, None,['not 100% strings', 134, 1.4, False],
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
                assert good_content == validation.validate_field_content(good_content, is_non_tensor_field=non_tensor_field)

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

        @mock.patch("os.environ", mock_environ)
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

    def test_index_name_validation(self):
        assert "my-index-name" == validation.validate_index_name("my-index-name")
        bad_names = ['.opendistro_security', 'security-auditlog-', 'security-auditlog-100']
        for n in bad_names:
            try:
                validation.validate_index_name(n)
                raise AssertionError
            except InvalidIndexNameError:
                pass

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
            "number_of_replicas":1
        }

    def test_validate_index_settings(self):

        good_settings =[
            {
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
            },
            {   # extra field in text_preprocessing: OK
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence",
                        "blah blah blah": "woohoo"
                    },
                    "image_preprocessing": {
                        "patch_method": None
                    }
                },
                "number_of_shards": 5,
                "number_of_replicas": 1
            },
            {  # extra field in image_preprocessing: OK
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence",
                    },
                    "image_preprocessing": {
                        "patch_method": None,
                        "blah blah blah": "woohoo"
                    }
                },
                "number_of_shards": 5,
                "number_of_replicas": 1
            }
        ]
        for settings in good_settings:
            assert settings == validation.validate_settings_object(settings)

    def test_validate_index_settings_model_properties(self):
        good_settings = self.get_good_index_settings()
        good_settings['index_defaults']['model_properties'] = dict()
        assert good_settings == validation.validate_settings_object(good_settings)

    def test_validate_index_settings_bad(self):
        bad_settings = [{
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": "2",
                    "split_overlap": "0",
                    "split_method": "sentence"
                },
                "image_preprocessing": {
                    "patch_method": None
                }
            },
            "number_of_shards": 5,
            "number_of_replicas" : -1
        },
        {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": "2",
                    "split_overlap": "0",
                    "split_method": "sentence"
                },
                "image_preprocessing": {
                    "patch_method": None
                }
            },
            "number_of_shards": 5
        },
        ]
        for bad_setting in bad_settings:
            try:
                validation.validate_settings_object(bad_setting)
                raise AssertionError
            except InvalidArgError as e:
                pass

    def test_validate_index_settings_missing_text_preprocessing(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validation.validate_settings_object(settings)
        del settings['index_defaults']['text_preprocessing']
        try:
            validation.validate_settings_object(settings)
            raise AssertionError
        except InvalidArgError:
            pass

    def test_validate_index_settings_missing_model(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validation.validate_settings_object(settings)
        del settings['index_defaults']['model']
        try:
            validation.validate_settings_object(settings)
            raise AssertionError
        except InvalidArgError:
            pass

    def test_validate_index_settings_missing_index_defaults(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validation.validate_settings_object(settings)
        del settings['index_defaults']
        try:
            validation.validate_settings_object(settings)
            raise AssertionError
        except InvalidArgError:
            pass

    def test_validate_index_settings_bad_number_shards(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validation.validate_settings_object(settings)
        settings['number_of_shards'] = -1
        try:
            validation.validate_settings_object(settings)
            raise AssertionError
        except InvalidArgError as e:
            pass

    def test_validate_index_settings_bad_number_replicas(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validation.validate_settings_object(settings)
        settings['number_of_replicas'] = -1
        try:
            validation.validate_settings_object(settings)
            raise AssertionError
        except InvalidArgError as e:
            pass

    def test_validate_index_settings_img_preprocessing(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validation.validate_settings_object(settings)
        settings['index_defaults']['image_preprocessing']["path_method"] = "frcnn"
        assert settings == validation.validate_settings_object(settings)

    def test_validate_index_settings_misplaced_fields(self):
        bad_settings = [
            {
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
                "model": "hf/all_datasets_v4_MiniLM-L6"  # model is also outside, here...
            },
            {
                "index_defaults": {
                    "image_preprocessing": {
                        "patch_method": None  # no models here
                    },
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_method": "sentence",
                        "split_overlap": 0
                    },
                    "treat_urls_and_pointers_as_images": False
                },
                "model": "open_clip/ViT-L-14/laion2b_s32b_b82k", # model here (bad)
                "number_of_shards": 5,
                "treat_urls_and_pointers_as_images": True
            },
            {
                "index_defaults": {
                    "image_preprocessing": {
                        "patch_method": None,
                        "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
                    },
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_method": "sentence",
                        "split_overlap": 0
                    },
                    "treat_urls_and_pointers_as_images": False,
                    "number_of_shards": 5,  # shouldn't be here
                },
                "treat_urls_and_pointers_as_images": True
            },
            {  # good, BUT extra field in index_defaults
                "index_defaults": {
                    "number_of_shards": 5,
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
                "number_of_shards": 5
            },
            {  # good, BUT extra field in root
                "model": "hf/all_datasets_v4_MiniLM-L6",
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
                "number_of_shards": 5
            }
        ]
        for bad_set in bad_settings:
            try:
                validation.validate_settings_object(bad_set)
                raise AssertionError
            except InvalidArgError as e:
                pass

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
            { # needs more nesting
                "type": "multimodal_combination",
                "weights": {
                    "some_text": 0.5
                }
            },
            {
                "my_combination_field": { # this dict is OK
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
        ]
        for mapping in mappings:
            try:
                validation.validate_mappings_object(mapping)
                raise AssertionError
            except InvalidArgError as e:
                pass

    def test_validate_multimodal_combination_object(self):
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
            assert d == validation.validate_multimodal_combination_object(d)

    def test_validate_multimodal_combination_object_invalid(self):
        mappings = [
            {
                "my_combination_field": { # valid mappings dir, but not valid multimodal
                    "type": "multimodal_combination",
                    "weights": {
                        "some_text": 0.5
                    }
                }
            },
            {
                "type": "othertype",  # bad type
                "weights": {
                    "some_text": 0.5

                }
            },
            {
                "type": "multimodal_combination",
                "non_weights": {  # unknown fieldname 'non_weights' config in multimodal_combination
                    "some_text": 0.5
                }
            },
            {
                "type": "multimodal_combination",
                # missing weights for multimodal_combination
            },
            {
                "type": "multimodal_combination",
                "weights": {"blah": "woo"}  # non-number weights
            },
            {
                "type": "multimodal_combination",
                "weights": {"blah": "1.3"}  # non-number weights
            },
            {
                "type": "multimodal_combination",
                "weights": {
                    "some_text": -4.6,
                    "other_text": 22
                },
                "extra_field": {"blah"}  # unknown field
            },
            {
                "type": "multimodal_combination",
                "weights": {
                    "some_text": -4.6,
                    "other_text": 22,
                    "nontext": True  # non-number
                },
            }
        ]
        for mapping in mappings:
            try:
                validation.validate_multimodal_combination_object(mapping)
                raise AssertionError
            except InvalidArgError as e:
                pass

    def test_validate_valid_context_object(self):
        valid_context_list = [
            {
                "tensor":[
                    {"vector" : [0.2132] * 512, "weight" : 0.32},
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
                    {"vector": [23,], "weight": 1},
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
            assert valid_context == validation.validate_context_object(valid_context)

    def test_validate_invalid_context_object(self):
        valid_context_list = [
            {
                # Typo in tensor
                "tensors": [
                    {"vector" : [0.2132] * 512, "weight" : 0.32},
                    {"vector": [0.2132] * 512, "weight": 0.32},
                    {"vector": [0.2132] * 512, "weight": 0.32},
                ]
            },
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
                    {"vector": [23,], "weight": 1},
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
                    {"vector" : str([0.2132] * 512), "weight": 0.32},
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
                validation.validate_context_object(invalid_context)
                raise AssertionError
            except InvalidArgError:
                pass

    def test_invalid_custom_score_fields(self):
        invalid_custom_score_fields_list = [
            {
                "multiply_scores_by":   #typo
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

        ]
        for invalid_custom_score_fields in invalid_custom_score_fields_list:
            try:
                validation.validate_custom_score_fields(invalid_custom_score_fields)
                raise AssertionError
            except InvalidArgError as e:
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
        ]

        for valid_custom_score_fields in valid_custom_score_fields_list:
            validation.validate_custom_score_fields(valid_custom_score_fields)