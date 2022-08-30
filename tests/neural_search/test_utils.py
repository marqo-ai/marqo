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

    def test_construct_authorized_url(self):
        assert "https://admin:admin@localhost:9200" == utils.construct_authorized_url(
            url_base="https://localhost:9200", username="admin", password="admin"
        )

    def test_construct_authorized_url_empty(self):
        assert "https://:@localhost:9200" == utils.construct_authorized_url(
            url_base="https://localhost:9200", username="", password=""
        )
