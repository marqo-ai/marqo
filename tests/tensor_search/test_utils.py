import json
import pprint
import unittest
from marqo.tensor_search import utils
from marqo.tensor_search import enums
from unittest import mock


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

    def test_contextualise_filter(self):
        expected_mappings = [
            ("(an_int:[0 TO 30] and an_int:2) AND abc:(some text)",
             f"({enums.TensorField.chunks}.an_int:[0 TO 30] and {enums.TensorField.chunks}.an_int:2) AND {enums.TensorField.chunks}.abc:(some text)")
        ]
        for given, expected in expected_mappings:
            assert expected == utils.contextualise_filter(
                given, simple_properties=["an_int", "abc"]
            )

    def test_check_device_is_available(self):
        mock_cuda_is_available = mock.MagicMock()
        mock_cuda_device_count = mock.MagicMock()
        for device_str, num_cuda_devices, expected in [
                    ("cpu", 0, True),
                    ("cpu", 3, True),
                    ("cuda", 1, True),
                    ("cuda", 0, False),
                    ("cuda:0", 1, True),
                    ("cuda:1", 2, True),
                    ("cuda:2", 2, False),
                ]:
            mock_cuda_is_available.return_value = True if num_cuda_devices > 0 else False
            mock_cuda_device_count.return_value = num_cuda_devices

            @mock.patch("torch.cuda.is_available", mock_cuda_is_available)
            @mock.patch("torch.cuda.device_count", mock_cuda_device_count)
            def run_test():
                assert expected == utils.check_device_is_available(device_str)
                return True
            assert run_test()

    def test_merge_dicts(self):
        base = {
            'lvl_0_a': {
                'lvl_1_a': 'efgh',
                'lvl_1_b': True
            },
            'lvl_0_b': 'abcd',
            'lvl_0_c': 1234,
            'lvl_0_d': ['abcdefgh']
        }
        base_hash = hash(json.dumps(base))
        preferences = {
            'lvl_0_a': {
                'lvl_1_b': False,
                'lvl_1_c': "abcabc"
            },
            'lvl_0_d': [{'lvl_0_d_1': 'cat dog'}],
            'lvl_0_e': {'lvl1_0_d_1': {'jump': {"skip": 'track'}}}
        }
        preferences_hash = hash(json.dumps(preferences))
        assert utils.merge_dicts(base, preferences) == {
            'lvl_0_a': {
                'lvl_1_a': 'efgh',
                'lvl_1_b': False,
                'lvl_1_c': "abcabc"
            },
            'lvl_0_b': 'abcd',
            'lvl_0_c': 1234,
            'lvl_0_d': [{'lvl_0_d_1': 'cat dog'}],
            'lvl_0_e': {'lvl1_0_d_1': {'jump': {"skip": 'track'}}}
        }
        # assert that they didn't mutate
        assert preferences_hash == hash(json.dumps(preferences))
        assert base_hash == hash(json.dumps(base))

    def test_merge_dicts_edge_cases(self):
        assert {} == utils.merge_dicts({}, {})
        assert {'abc': '123', "zzz": {"wow": "cool"}} \
               == utils.merge_dicts({'abc': '123', "zzz": {"wow": "cool"}}, {})
        assert {'abc': '123', "zzz": {"wow": "cool"}} \
               == utils.merge_dicts({}, {'abc': '123', "zzz": {"wow": "cool"}})
        assert {'abc': '123', "zzz": {"wow": "cool"}} \
               == utils.merge_dicts({'zzz': {"wow": "rough"}}, {'abc': '123', "zzz": {"wow": "cool"}})

    def test_merge_nones(self):
        base = {
            'lvl_0_a': {
                'lvl_1_a': 'efgh',
                'lvl_1_b': True
            },
            'lvl_0_b': 'abcd',
            'lvl_0_c': 1234,
            'lvl_0_d': ['abcdefgh'],
            'lvl_0_e': {'lvl1_0_d_1': {'jump': {"skip": 'track'}}}
        }
        base_hash = hash(json.dumps(base))
        preferences = {
            'lvl_0_a': {
                'lvl_1_b': None,
                'lvl_1_c': "abcabc"
            },
            'lvl_0_d': [{'lvl_0_d_1': 'cat dog'}],
            'lvl_0_e': None
        }
        preferences_hash = hash(json.dumps(preferences))
        assert utils.merge_dicts(base, preferences) == {
            'lvl_0_a': {
                'lvl_1_a': 'efgh',
                'lvl_1_b': True,
                'lvl_1_c': "abcabc"
            },
            'lvl_0_b': 'abcd',
            'lvl_0_c': 1234,
            'lvl_0_d': [{'lvl_0_d_1': 'cat dog'}],
            'lvl_0_e': {'lvl1_0_d_1': {'jump': {"skip": 'track'}}}
        }
        # assert that they didn't mutate
        assert preferences_hash == hash(json.dumps(preferences))
        assert base_hash == hash(json.dumps(base))

    def test_read_env_vars_and_defaults(self):
        """Make sure the priority order is expected
        (environment vars > defaults else None) """
        for key, mock_real_environ, default_vars, expected in [
            ("SOME_VAR", dict(), dict(), None),
            ("SOME_VAR", {"SOME_VAR": "1234"}, dict(), "1234"),
            ("SOME_VAR", dict(), {"SOME_VAR": "1234"}, "1234"),
            ("SOME_VAR", {"SOME_VAR": "111"}, {"SOME_VAR": "333"}, "111"),
        ]:
            mock_default_env_vars = mock.MagicMock()
            mock_default_env_vars.return_value = default_vars

            @mock.patch("marqo.tensor_search.configs.default_env_vars", mock_default_env_vars)
            @mock.patch("os.environ", mock_real_environ)
            def run():
                assert expected == utils.read_env_vars_and_defaults(var=key)
                return True
            assert run()
