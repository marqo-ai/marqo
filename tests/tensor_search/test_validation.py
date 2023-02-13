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
            {123}, [], None, {"abw": "cjnk"}
        ]
        for bad_content in bad_field_content:
            try:
                validation.validate_field_content(bad_content)
                raise AssertionError
            except InvalidArgError as e:
                pass

    def test_validate_field_content_good(self):
        good_field_content = [
            123, "heehee", 12.4, False
        ]
        for good_content in good_field_content:
            assert good_content == validation.validate_field_content(good_content)

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



