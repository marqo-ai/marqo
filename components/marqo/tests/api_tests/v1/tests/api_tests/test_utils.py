import unittest

from marqo import utils


class TestUtils(unittest.TestCase):

    def test_construct_authorized_url(self):
        assert "https://admin:admin@localhost:9200" == utils.construct_authorized_url(
            url_base="https://localhost:9200", username="admin", password="admin"
        )

    def test_construct_authorized_url_empty(self):
        assert "https://:@localhost:9200" == utils.construct_authorized_url(
            url_base="https://localhost:9200", username="", password=""
        )

    def test_translate_device_string_for_url(self):
        translations = [("cpu", "cpu"), ("CUDA", "cuda"), (None, None),
                        ("cpu:1", "cpu1"), ("CUDA:24", "cuda24"),
                        ("cuda2", "cuda2")]
        for to_be_translated, expected in translations:
            assert expected == utils.translate_device_string_for_url(to_be_translated)