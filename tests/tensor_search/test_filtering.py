import json
import os
import pathlib
import pprint
import unittest
from marqo.tensor_search import filtering
from marqo.tensor_search import enums
from marqo import errors
from unittest import mock
from unittest.mock import patch, MagicMock


class TestFiltering(unittest.TestCase):
    def test_contextualise_user_filter(self):
        expected_mappings = [
            (   # fields with no spaces
                "(an_int:[0 TO 30] AND an_int:2) AND abc:(some text)",
                ["an_int", "abc"],
                f"({enums.TensorField.chunks}.an_int:[0 TO 30] AND {enums.TensorField.chunks}.an_int:2) AND {enums.TensorField.chunks}.abc:(some text)"
            ),
            (   # fields with spaces
                r'spaced\ int:[0 TO 30]',
                ["spaced int"],
                rf'{enums.TensorField.chunks}.spaced\ int:[0 TO 30]'
            ),
            (   # field in string not in properties
                "field_not_in_properties:random AND normal_field:3",
                ["normal_field"],
                f"field_not_in_properties:random AND {enums.TensorField.chunks}.normal_field:3"
            ),
            (   # properties has field not in string
                "normal_field:3",
                ["normal_field", "field_not_in_string"],
                f"{enums.TensorField.chunks}.normal_field:3"
            ),
            (
                None,
                ["random_field_1", "random_field_2"],
                ""
            ),
            (
                "",
                ["random_field_1", "random_field_2"],
                ""
            )
        ]
        for given_filter_string, given_simple_properties, expected in expected_mappings:
            assert expected == filtering.contextualise_user_filter(
                filter_string=given_filter_string, 
                simple_properties=given_simple_properties,
            )
    
    def test_build_searchable_attributes_filter(self):
        expected_mappings = [
            (["an_int", "abc"],
             f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc) OR {enums.TensorField.chunks}.{enums.TensorField.field_name}:(an_int)"),
            (["an_int"],
             f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(an_int)"),
            ([], "")
        ]
        for given, expected in expected_mappings:
            assert expected == filtering.build_searchable_attributes_filter(
                given
            )
    
    def test_build_tensor_search_filter(self):
        test_cases = (
            {
                "filter_string": "abc:(some text)",
                "simple_properties": ["abc"],
                "searchable_attributes": ["bleh"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(bleh)) AND ({enums.TensorField.chunks}.abc:(some text))"
            },
            # special character in searchable attribute
            # escaped space in filter string
        )
        for case in test_cases:
            assert case["expected"] == filtering.build_tensor_search_filter(
                filter_string=case["filter_string"],
                simple_properties=case["simple_properties"],
                searchable_attribs=case["searchable_attributes"]
            )
    
    def test_sanitise_lucene_special_chars(self):
        expected_mappings = [
            ("plain text", "plain text"),
            ("exclamation!", "exclamation\\!"),
            ("some text?", "some text\\?"),
            ("some text&&", "some text\\&&"),
            ("some text\\", "some text\\\\"),
            ("everything ||&?\\ combined", "everything \\||&\\?\\\\ combined"),
            ("some text&", "some text&")        # no change, & is not a special char
        ]
        for given, expected in expected_mappings:
            assert expected == filtering.sanitise_lucene_special_chars(
                given
            )