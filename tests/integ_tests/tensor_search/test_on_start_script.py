from unittest import mock

from integ_tests.marqo_test import MarqoTestCase
from marqo.tensor_search import on_start_script


class TestOnStartScript(MarqoTestCase):

    @mock.patch("marqo.config.Config")
    def test_boostrap_failure_should_raise_error(self, mock_config):
        mock_config.index_management.bootstrap_vespa.side_effect = Exception('some error')

        with self.assertRaises(Exception) as context:
            on_start_script.on_start(mock_config)

        self.assertTrue('some error' in str(context.exception))
