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
    def test_add_chunks_prefix_to_filter_string_fields(self):
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
                "field\\&&\\||withspecialchars:(random \\+value)",
                ["field&&||withspecialchars"],
                f"{enums.TensorField.chunks}.field\\&&\\||withspecialchars:(random \\+value)"
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
            (   # field is not at start and is ending substring of another field with space or parenthesis before it
                "random:random OR field_a:a AND another\\ field_a:b",
                ["field_a", "another field_a"],
                f"random:random OR {enums.TensorField.chunks}.field_a:a AND {enums.TensorField.chunks}.another\\ field_a:b"
            ),
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
            (
                # nested parenthesis
                "(field_a:(inner_a AND inner_b) OR field_b:outer_b) AND field_c:c", 
                ["field_a", "field_b", "field_c"], 
                f"({enums.TensorField.chunks}.field_a:(inner_a AND inner_b) OR {enums.TensorField.chunks}.field_b:outer_b) AND {enums.TensorField.chunks}.field_c:c"
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
            ),
            (   # empty properties
                "random_field_1:random AND random_field_2:random",
                [],
                "random_field_1:random AND random_field_2:random"
            ),
            (   # empty both
                "",
                [],
                ""
            )
        ]
        for given_filter_string, given_simple_properties, expected in expected_mappings:
            prefixed_filter_string = filtering.add_chunks_prefix_to_filter_string_fields(
                filter_string=given_filter_string,
                simple_properties=given_simple_properties,
            )
            assert expected == prefixed_filter_string

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
            (["field(with)parenthesis"],
                f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(field\\(with\\)parenthesis)"),
            # searchable attribute with special characters in it
            (["field\\with&&special+characters"],
                f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(field\\\\with\\&&special\\+characters)"),
            # multiple searchable attribute with special characters in it
            (["field\\with&&special+characters", "another_field\\with&&special+characters"],
                f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(another_field\\\\with\\&&special\\+characters) OR {enums.TensorField.chunks}.{enums.TensorField.field_name}:(field\\\\with\\&&special\\+characters)"),
            ([], ""),
            (None, "")
        ]
        for given, expected in expected_mappings:
            assert expected == filtering.build_searchable_attributes_filter(
                given
            )
    
    def test_build_tensor_search_filter(self):
        # Format notes:
        # searchable attributes filter will come BEFORE the user filter
        # searchable attributes will be inserted backwards
        test_cases = (
            {
                "filter_string": "abc:(some text)",
                "simple_properties": {"abc": {'type': 'text'}},
                "searchable_attributes": ["abc"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc)) AND ({enums.TensorField.chunks}.abc:(some text))"
            },
            { # parenthesis in searchable attribute
                "filter_string": "abc:(some text)",
                "simple_properties": {"abc": {'type': 'text'}},
                "searchable_attributes": ["abc(with)parenthesis"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc\\(with\\)parenthesis)) AND ({enums.TensorField.chunks}.abc:(some text))"
            },
            
            { # empty searchable attributes
                "filter_string": "abc:(some text)",
                "simple_properties": {"abc": {'type': 'text'}},
                "searchable_attributes": [],
                "expected": f"{enums.TensorField.chunks}.abc:(some text)"
            },
            { # None searchable attributes
                "filter_string": "abc:(some text)",
                "simple_properties": {"abc": {'type': 'text'}},
                "searchable_attributes": None,
                "expected": f"{enums.TensorField.chunks}.abc:(some text)"
            },
            { # parenthesis in filter string (escaped)
                "filter_string": "abc\\(:(some te\\)xt)",
                "simple_properties": {"abc(": {'type': 'text'}},
                "searchable_attributes": ["def"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(def)) AND ({enums.TensorField.chunks}.abc\\(:(some te\\)xt))"
            },
            { # : in searchable attribute and filter string
                "filter_string": "colon\\:here:(some text)",
                "simple_properties": {"colon:here": {'type': 'text'}},
                "searchable_attributes": ["colon:here:also"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(colon\\:here\\:also)) AND ({enums.TensorField.chunks}.colon\\:here:(some text))"
            },
            { # filter string containing 'AND' operator
                "filter_string": "abc:(some text) AND def:(another text)",
                "simple_properties": {"abc": {'type': 'text'}, "def": {'type': 'text'}},
                "searchable_attributes": ["abc", "def"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(def) OR {enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc)) AND ({enums.TensorField.chunks}.abc:(some text) AND {enums.TensorField.chunks}.def:(another text))"
            },
            { # filter string containing 'OR' operator
                "filter_string": "abc:(some text) OR def:(another text)",
                "simple_properties": {"abc": {'type': 'text'}, "def": {'type': 'text'}},
                "searchable_attributes": ["abc", "def"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(def) OR {enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc)) AND ({enums.TensorField.chunks}.abc:(some text) OR {enums.TensorField.chunks}.def:(another text))"
            },
            { # filter string containing 'NOT' operator
                "filter_string": "abc:(some text) NOT def:(another text)",
                "simple_properties": {"abc": {'type': 'text'}, "def": {'type': 'text'}},
                "searchable_attributes": ["abc", "def"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(def) OR {enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc)) AND ({enums.TensorField.chunks}.abc:(some text) NOT {enums.TensorField.chunks}.def:(another text))"
            },
            { # filter string with escaped spaces
                "filter_string": "abc\\ :(some text)",
                "simple_properties": {"abc ": {'type': 'text'}},
                "searchable_attributes": ["def"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(def)) AND ({enums.TensorField.chunks}.abc\\ :(some text))"
            },
            { # filter string with special chars that have different meanings when escaped or unescaped
                "filter_string": "abc\\ \\+:(some text)",
                "simple_properties": {"abc +": {'type': 'text'}},
                "searchable_attributes": ["def"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(def)) AND ({enums.TensorField.chunks}.abc\\ \\+:(some text))"
            },
            { # filter string with multiple properties (no operation specified, OpenSearch will use OR)
                "filter_string": "abc:(some text) def:(another text)",
                "simple_properties": {"abc": {'type': 'text'}, "def": {'type': 'text'}},
                "searchable_attributes": ["wx", "yz"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(yz) OR {enums.TensorField.chunks}.{enums.TensorField.field_name}:(wx)) AND ({enums.TensorField.chunks}.abc:(some text) {enums.TensorField.chunks}.def:(another text))"
            },
            { # empty filter string
                "filter_string": "",
                "simple_properties": {"abc": {'type': 'text'}},
                "searchable_attributes": ["def"],
                "expected": f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(def)"
            },
            { # None filter string
                "filter_string": None,
                "simple_properties": {"abc": {'type': 'text'}},
                "searchable_attributes": ["def"],
                "expected": f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(def)"
            },
            { # empty simple properties
                "filter_string": "abc:(some text)",     # chunks prefix will NOT be added
                "simple_properties": {},
                "searchable_attributes": ["def"],
                "expected": f"({enums.TensorField.chunks}.{enums.TensorField.field_name}:(def)) AND (abc:(some text))"
            },
            { # None simple properties
                "filter_string": "abc:(some text)",     # chunks prefix will NOT be added
                "simple_properties": None,
                "searchable_attributes": ["def"],
                "expected": errors.InternalError
            },
            { # empty all
                "filter_string": "",
                "simple_properties": {},
                "searchable_attributes": [],
                "expected": ""
            }
        )
        for case in test_cases:
            try:
                tensor_search_filter = filtering.build_tensor_search_filter(
                    filter_string=case["filter_string"],
                    simple_properties=case["simple_properties"],
                    searchable_attribs=case["searchable_attributes"]
                )
                assert case["expected"] == tensor_search_filter
            except case["expected"]:
                # expected will be a specific error, if the case should fail
                pass

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