import json
import os
import pathlib
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

    def test_parse_lexical_query(self):
        # 2-tuples of input text, and expected parse_lexical_query() output
        cases = [
            ('just a string', ([], 'just a string')),
            ('just a "string"', (["string"], 'just a')),
            ('just "a" string', (["a"], 'just string')),
            ('"just" a string', (["just"], 'a string')),
            ('just "a long long " string', (["a long long "], 'just string')),
            ('"required 1 " not required " required2" again', (["required 1 ", " required2"], 'not required again')),
            ('"just" "just" "" a string', (["just", "just", ""], 'a string')),

            ('朋友你好', ([], '朋友你好')),
            ('朋友 "你好"', (["你好"], '朋友')),
            # spaces get introduced, even though Chinese doesn't use them:
            ('你好 "老" 朋友', (["老"], '你好 朋友')),
            ('"朋友" 你好', (["朋友"], '你好')),

            ('', ([], '')),
            ('"cookie"', (["cookie"], '')),
            ('"朋友"', (["朋友"], '')),
            ('"', ([], '"')),
            ('"""hello', ([], '"""hello')),
            ('""" python docstring appeared"""', ([], '""" python docstring appeared"""')),
            ('""', ([''], '')),
            ('what about backticks `?', ([], 'what about backticks `?')),
            ('\\" escaped quotes\\"  what happens here?', ([], '" escaped quotes" what happens here?')),
            ('\\"朋友\\"', ([], '"朋友"')),
            ('double  spaces  get  removed', ([], 'double spaces get removed')),
            ('"go"od"', ([], '"go"od"')),
            ('"ter"m1" term2', ([], '"ter"m1" term2')),
            ('"term1" "term2" "term3', ([], '"term1" "term2" "term3')),
            ('"good', ([], '"good')),
            ('"朋友', ([], '"朋友')),

            # on Lucene, these unusual structures seem to get passed straight through as well.
            # The quotes seem to be completely ignored (with and without quotes yields identical results,
            # including scores):
            ('"go"od" a"', ([], '"go"od" a"')),
            ('"sam"a', ([], '"sam"a')),
            ('"sam"?', ([], '"sam"?')),
            ('"朋友"你好', ([], '"朋友"你好')),

        ]
        for input, expected_output in cases:
            assert utils.parse_lexical_query(input) == expected_output

    def test_parse_lexical_query_non_string(self):
        non_strings = [124, None, 1.4, False, dict(), [1, 2]]

        for ns in non_strings:
            try:
                utils.parse_lexical_query(ns)
                raise AssertionError
            except TypeError as e:
                assert "string as input" in str(e)

    def test_get_marqo_root_returns_str(self):
        self.assertIsInstance(utils._get_marqo_root(), str)

    def test_get_marqo_root_returns_correct_path(self):
        assert utils._get_marqo_root().endswith('marqo/src/marqo')

    def test_get_marqo_root_returns_existing_path(self):
        assert os.path.exists(utils._get_marqo_root())

    def test_get_marqo_root_cwd_agnostic(self):
        original_dir = os.getcwd()
        original_marqo_root = utils._get_marqo_root()
        # change to new dir (home dir)
        os.chdir(os.path.dirname(pathlib.Path.home()))
        new_dir = os.getcwd()
        # ensure we are in a different place
        assert str(new_dir) != str(original_dir)
        # marqo_root should still be equal:
        self.assertEqual(utils._get_marqo_root(), original_marqo_root)
        # reset cwd
        os.chdir(original_dir)

    def test_get_marqo_root_from_env_returns_str(self):
        self.assertIsInstance(utils.get_marqo_root_from_env(), str)

    def test_get_marqo_root_from_env_returns_correct_path(self):
        assert utils.get_marqo_root_from_env().endswith('marqo/src/marqo')

    def test_get_marqo_root_from_env_returns_existing_path(self):
        assert os.path.exists(utils.get_marqo_root_from_env())

    def test_get_marqo_root_from_env_cwd_agnostic(self):
        original_dir = os.getcwd()
        original_marqo_root = utils.get_marqo_root_from_env()
        # change to new dir (home dir)
        os.chdir(os.path.dirname(pathlib.Path.home()))
        new_dir = os.getcwd()
        # ensure we are in a different place
        assert str(new_dir) != str(original_dir)
        # marqo_root should still be equal:
        self.assertEqual(utils.get_marqo_root_from_env(), original_marqo_root)
        # reset cwd
        os.chdir(original_dir)

    def test_get_marqo_root_from_env_returns_env_var_if_exists(self):
        expected = "/Users/CoolUser/marqo/src/marqo"
        with mock.patch.dict('os.environ', {enums.EnvVars.MARQO_ROOT_PATH: expected}):
            actual = utils.get_marqo_root_from_env()
        self.assertEqual(actual, expected)

    def test_creates_env_var_if_not_exists(self):
        @mock.patch("os.environ", dict())
        def run():
            assert enums.EnvVars.MARQO_ROOT_PATH not in os.environ
            marqo_root = utils.get_marqo_root_from_env()
            assert marqo_root.endswith('marqo/src/marqo')
            assert os.environ[enums.EnvVars.MARQO_ROOT_PATH] == marqo_root
            return True
        assert run()

    def test_generate_batches_batch_size(self):
        # Test that each batch has the correct size
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k = 3
        batches = list(utils.generate_batches(seq, batch_size=k))
        self.assertEqual(len(batches), 4)
        for batch in batches[:-1]:
            self.assertEqual(len(batch), k)
        self.assertEqual(len(batches[-1]), 1)

    def test_generate_batches_batch_contents(self):
        # Test that each batch contains the correct elements
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k = 3
        batches = list(utils.generate_batches(seq, batch_size=k))
        expected_batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        for i, batch in enumerate(batches):
            self.assertEqual(batch, expected_batches[i])

    def test_generate_batches_empty_sequence(self):
        # Test that an empty sequence returns an empty generator
        seq = []
        k = 3
        batches = list(utils.generate_batches(seq, batch_size=k))
        self.assertEqual(len(batches), 0)

    def test_generate_batches_0_batch_size(self):
        # Test that an empty sequence returns an empty generator
        seq = [1, 2]
        try:
            batches = list(utils.generate_batches(seq, batch_size=0))
            raise AssertionError
        except ValueError as e:
            assert "must be greater than 0" in str(e)

    def test_generate_batches_0_batch_size_empty_seq(self):
        # Test that an empty sequence returns an empty generator
        seq = []
        try:
            batches = list(utils.generate_batches(seq, batch_size=0))
            raise AssertionError
        except ValueError as e:
            assert "must be greater than 0" in str(e)

