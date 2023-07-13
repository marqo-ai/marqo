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
            (   # multiple fields with no spaces
                "(an_int:[0 TO 30] AND an_int:2) AND abc:(some text)",
                ["an_int", "abc"],
                f"({enums.TensorField.chunks}.an_int:[0 TO 30] AND {enums.TensorField.chunks}.an_int:2) AND {enums.TensorField.chunks}.abc:(some text)"
            ),
            (   # fields with spaces
                "spaced\\ int:[0 TO 30]",
                ["spaced int"],
                f"{enums.TensorField.chunks}.spaced\\ int:[0 TO 30]"
            ),
            (   # fields with special chars
                "field\\&&\\||withspecialchars:(random value)",
                ["field&&||withspecialchars"],
                f"{enums.TensorField.chunks}.field\\&&\\||withspecialchars:(random value)"
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
            (   # field is substring of start of filter string, should be ignored
                "field_field_field:foo",
                ["field"],
                "field_field_field:foo"
            ),
            (   # field name at start
                "field_field_field:foo",
                ["field_field_field"],
                f"{enums.TensorField.chunks}.field_field_field:foo"
            ),
            (   # field name at start with parenthesis
                "(field_a:a OR field_b:b) AND field_c:c",
                ["field_a"],
                f"({enums.TensorField.chunks}.field_a:a OR field_b:b) AND field_c:c"
            ),
            (   # field name at start and has : in it
                "field\\:a:a",
                ["field:a"],
                f"{enums.TensorField.chunks}.field\\:a:a"
            ),
            (   # field is at start and is ending substring of another field
                "field_a:a AND another_field_a:b",
                ["field_a", "another_field_a"],
                f"{enums.TensorField.chunks}.field_a:a AND {enums.TensorField.chunks}.another_field_a:b"
            ),
            # we may need to make out own tokenizer to fix this edge case.
            #(   # field is not at start and is ending substring of another field with space or parenthesis before it
            #    "random:random OR field_a:a AND another\\ field_a:b",
            #    ["field_a", "another field_a"],
            #    f"random:random OR {enums.TensorField.chunks}.field_a:a AND {enums.TensorField.chunks}.another\\ field_a:b"
            #),
            (   # field name in the middle and has : in it
                "random:random OR field\\:a:a",
                ["field:a"],
                f"random:random OR {enums.TensorField.chunks}.field\\:a:a"
            ),
            (   # field appears multiple times in filter string
                "field_a:a AND field_b:b OR (field_a:c OR field_a:d)",
                ["field_a"],
                f"{enums.TensorField.chunks}.field_a:a AND field_b:b OR ({enums.TensorField.chunks}.field_a:c OR {enums.TensorField.chunks}.field_a:d)"
            ),
            
            (   # field is substring of another field
                "field_a:a AND field_a_another:b",
                ["field_a", "field_a_another"],
                f"{enums.TensorField.chunks}.field_a:a AND {enums.TensorField.chunks}.field_a_another:b"
            ),
            (   # field name in the middle
                "field_a:a AND field_b:b OR field_c:c",
                ["field_b"],
                f"field_a:a AND {enums.TensorField.chunks}.field_b:b OR field_c:c"
            ),
            (   # field name in the middle with parenthesis
                "field_a:a AND (field_b:b OR field_c:c)",
                ["field_b"],
                f"field_a:a AND ({enums.TensorField.chunks}.field_b:b OR field_c:c)"
            ),
            (   # content has field name in it
                "field_a:field_a",
                ["field_a"],
                f"{enums.TensorField.chunks}.field_a:field_a"
            ),
            (   # None filter string
                None,
                ["random_field_1", "random_field_2"],
                ""
            ),
            (   # empty filter string
                "",
                ["random_field_1", "random_field_2"],
                ""
            )
        ]
        for given_filter_string, given_simple_properties, expected in expected_mappings:
            contextualised_user_filter = filtering.contextualise_user_filter(
                filter_string=given_filter_string,
                simple_properties=given_simple_properties,
            )
            assert expected == contextualised_user_filter

    def test_build_searchable_attributes_filter(self):
        expected_mappings = [
            # multiple searchable attributes
            (["an_int", "abc"],
             f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc) OR {enums.TensorField.chunks}.{enums.TensorField.field_name}:(an_int)"),
            # single searchable attribute
            (["an_int"],
             f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(an_int)"),
            # searchable attribute with space in it
            (["field with spaces"],
                f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(field\\ with\\ spaces)"),
            # searchable attribute with : in it
            (["field:with:colons"],
                f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(field\\:with\\:colons)"),
            # searchable attribute with parenthesis in it
            # searchable attribute with special characters in it
            ([], ""),
            (None, "")
        ]
        for given, expected in expected_mappings:
            assert expected == filtering.build_searchable_attributes_filter(
                given
            )
    
    def test_build_tensor_search_filter(self):
        test_cases = (
            {
                "filter_string": "abc:(some text)",
                "simple_properties": {"abc": "xyz"},
                "searchable_attributes": ["abc"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc)) AND ({enums.TensorField.chunks}.abc:(some text))"
            },
            # parenthesis in searchable attribute
            # empty searchable attributes
            # None searchable attributes
            # parenthesis in filter string (escaped)
            # empty filter string
            # None filter string
            # : in searchable attribute and filter string
            # empty simple properties
            # None simple properties
            # empty all
        )
        for case in test_cases:
            tensor_search_filter = filtering.build_tensor_search_filter(
                filter_string=case["filter_string"],
                simple_properties=case["simple_properties"],
                searchable_attribs=case["searchable_attributes"]
            )
            assert case["expected"] == tensor_search_filter

    def test_sanitise_lucene_special_chars(self):
        expected_mappings = [
            ("text with space", "text\\ with\\ space"),
            ("exclamation!", "exclamation\\!"),
            ("sometext?", "sometext\\?"),
            ("sometext&&", "sometext\\&&"),
            ("sometext\\", "sometext\\\\"),
            ("everything ||&?\\combined", "everything\\ \\||&\\?\\\\combined"),
            ("sometext&", "sometext&")        # no change, & is not a special char
        ]
        for given, expected in expected_mappings:
            assert expected == filtering.sanitise_lucene_special_chars(
                given
            )