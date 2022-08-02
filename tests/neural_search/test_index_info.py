import pprint
import unittest
from marqo.neural_search.models.index_info import IndexInfo
from marqo.neural_search.models import index_info
from marqo.neural_search.enums import NeuralField
from marqo.neural_search import configs


class TestIndexInfo(unittest.TestCase):

    def test_get_vector_properties_empty(self):
        """This shouldn't happen, because there would at least be __field_name from
        index creation"""
        ii = IndexInfo(model_name='a', properties=dict(),
                       neural_settings=configs.get_default_neural_index_settings())
        try:
            ii.get_vector_properties()
            raise AssertionError
        except KeyError as e:
            assert NeuralField.chunks in str(e)

    def test_get_text_properties_empty(self):
        """Text properties aren't nested, so it handles empty properties fine. (no KeyError)"""
        ii = IndexInfo(model_name='a', properties=dict(),
                       neural_settings=configs.get_default_neural_index_settings())
        assert dict() == ii.get_text_properties()

    def test_get_vector_properties(self):
        ii = IndexInfo(
            model_name='a',
            properties={
                "a": {1: 2}, "b": {1: 2},
                NeuralField.chunks: {"properties":{
                    "__vector_a": {1: 2},
                    NeuralField.field_name: {'a': 'b'}, NeuralField.field_content: {"a": "b"}}}},
            neural_settings=configs.get_default_neural_index_settings()
        )
        assert {"__vector_a": {1: 2}} == ii.get_vector_properties()

    def test_get_vector_properties_tricky_names(self):
        ii = IndexInfo(
            model_name='a', properties={
                "a": {1: 2},
                NeuralField.chunks: {"properties": {
                    NeuralField.field_name: {'a': 'b'},
                    NeuralField.field_content: {"a": "b"},
                    "__vector_a": {1: 2}, "__vector_Some title": {1: 2},
                }}
            }, neural_settings=configs.get_default_neural_index_settings()
        )
        assert {"__vector_a": {1: 2},
                "__vector_Some title": {1: 2}} == ii.get_vector_properties()

    def test_get_vector_properties_no_vectors(self):
        ii = IndexInfo(model_name='a', properties={
            "a": {1: 2}, "b_a": {1: 2}, "blah blah": {1: 2},
            NeuralField.chunks: {"properties": {
                NeuralField.field_name: {'a': 'b'},
                NeuralField.field_content: {"a": "b"},
            }}
        }, neural_settings=configs.get_default_neural_index_settings())
        assert dict() == ii.get_vector_properties()

    def test_get_text_properties(self):
        ii = IndexInfo(
            model_name='a',
            properties={
                "a": {1: 2}, "b_a": {1: 2}, "blah blah": {1: 2},
                NeuralField.chunks: {"properties": {
                    NeuralField.field_name: {'a': 'b'},
                    NeuralField.field_content: {"a": "b"},
                }}
            },
            neural_settings=configs.get_default_neural_index_settings()
       )
        assert {"a": {1: 2}, "blah blah": {1: 2},
                "b_a": {1: 2}} == ii.get_text_properties()

    def test_get_text_properties_no_text_props(self):
        ii = IndexInfo(
            model_name='some model',
            properties={
                "__vector_a": {1: 2}, "__vector_b_a": {1: 2},  "__vector_blah blah": {1: 2},
                "__field_name": {1:2},
                NeuralField.chunks: {NeuralField.field_name: {'a': 'b'}, NeuralField.field_content: {"a": "b"}}
            }, neural_settings=configs.get_default_neural_index_settings())
        assert dict() == ii.get_text_properties()

    def test_get_text_properties_some_text_props(self):
        ii = IndexInfo(
            model_name='some model',
            properties={
            "__vector_a": {1: 2}, "__vector_b_a": {1: 2},  "__vector_blah blah": {1: 2},
            "__field_name": {1: 2}, "some_text_prop": {1:2334}, "cat": {"hat": "ter"},
            "__doc_chunk_relation": {"afafa": "afafa"},
            NeuralField.chunks: {NeuralField.field_name: {'a': 'b'}, NeuralField.field_content: {"a": "b"}}
            },
            neural_settings=configs.get_default_neural_index_settings()
        )
        assert {"some_text_prop": {1:2334}, "cat": {"hat": "ter"}} == ii.get_text_properties()
