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
    def test_contextualise_filter(self):
        expected_mappings = [
            ("(an_int:[0 TO 30] and an_int:2) AND abc:(some text)",
             f"({enums.TensorField.chunks}.an_int:[0 TO 30] and {enums.TensorField.chunks}.an_int:2) AND {enums.TensorField.chunks}.abc:(some text)")
            # TODO: Add more edge cases / special chars
        ]
        for given, expected in expected_mappings:
            assert expected == filtering.contextualise_user_filter(
                given, simple_properties=["an_int", "abc"]
            )
    
    def test_build_searchable_attributes_filter(self):
        expected_mappings = [
            (["an_int", "abc"],
             f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(abc) OR {enums.TensorField.chunks}.{enums.TensorField.field_name}:(an_int)"),
            (["an_int"],
             f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:(an_int)")
            # TODO: Add more edge cases / special chars
        ]
        for given, expected in expected_mappings:
            assert expected == filtering.build_searchable_attributes_filter(
                given
            )
    
    def test_build_tensor_search_filter(self):
        # TODO: test edge cases
        pass
    
    def test_sanitise_lucene_special_chars(self):
        expected_mappings = [
            ("some text", "some text"),
            ("some text!", "some text\!"),
            ("some text?", "some text\?"),
            ("some text&&", "some text\&&")
            # TODO: fix and add better cases
        ]
        for given, expected in expected_mappings:
            assert expected == filtering.sanitise_lucene_special_chars(
                given
            )