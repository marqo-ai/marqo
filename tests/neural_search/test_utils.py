import pprint
import unittest
from marqo.neural_search import utils

class TestUtils(unittest.TestCase):
 
    def test__reduce_vectors(self):
        assert {
                "__vector_abc": [1,2,3]
            } == utils.truncate_dict_vectors({
                "__vector_abc": [1,2,3,4,5,6,7,8]
            }, new_length=3)

    def test__reduce_vectors_nested(self):
        assert {
                  "vs": [{"otherfield": "jkerhjbrbhj", "__vector_abc": [1, 2, 3]}]
            } == utils.truncate_dict_vectors({
                "vs": [{"otherfield": "jkerhjbrbhj", "__vector_abc": [1,2,3,4,5,6,7,8]}]
        }, new_length=3)