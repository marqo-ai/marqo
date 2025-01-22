import io
import math
import os
import tempfile
import unittest
import yaml
import docker
from xml.etree import ElementTree as ET
from xml.dom import minidom
from unittest.mock import patch, mock_open, call
from tests.marqo_test import MarqoTestCase
from scripts.vespa_local.vespa_local import VespaLocalSingleNode, VespaLocalMultiNode
import builtins


class TestVespaLocal(MarqoTestCase):
    def setUp(self):
        # Create a temporary directory and switch to it
        self.test_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir.name)
        # Ensure multinode directory exists for file writes.
        os.makedirs("multinode", exist_ok=True)

        # Create a dedicated mock for write operations.
        self.write_mock = mock_open()
        self.real_open = builtins.open  # Save the unpatched builtins.open

        self.test_cases = [
            (1, 1),
            (2, 0),
            (2, 1)
        ]

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.test_dir.cleanup()

    def custom_open(self, path: str, mode: str, *args, **kwargs):
        """
        If mode is for reading, use the real open,
        otherwise use a mock open.
        """
        if 'r' in mode and 'w' not in mode:
            return self.real_open(path, mode, *args, **kwargs)
        else:
            # For write mode, we'll use our global mock_open provided from the patch.
            return self.write_mock(path, mode, *args, **kwargs)

    def _read_file(self, path: str) -> str:
        currentdir = os.path.dirname(os.path.abspath(__file__))
        abspath = os.path.join(currentdir, path)

        with open(abspath, 'r') as f:
            file_content = f.read()

        return file_content


class TestVespaLocalMultiNode(TestVespaLocal):
    @patch("builtins.open", side_effect=lambda path, mode, *args, **kwargs: None)
    def test_generate_docker_compose(self, patched_open):
        VESPA_VERSION = "8.431.32"
        # Patch file write (to check content) but use original open for reading.
        patched_open.side_effect = self.custom_open
        for number_of_shards, number_of_replicas in self.test_cases:
            with self.subTest(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas):
                VespaLocalMultiNode.generate_docker_compose(number_of_shards, number_of_replicas, VESPA_VERSION)

                # Verify that docker-compose.yml is written
                patched_open.assert_any_call('multinode/docker-compose.yml', 'w')

                handle = self.write_mock()
                written_yml = "".join([call_arg[0][0] for call_arg in handle.write.call_args_list])

                # Check that written YML exactly matches the expected YML
                expected_file_name = f"expected/docker-compose_{number_of_shards}_shard_{number_of_replicas}_replica.yml"
                expected_yml = self._read_file(expected_file_name)
                self.assertEqual(written_yml, expected_yml)

                # Reset call_args_list to avoid tests failing due to previous calls
                self.write_mock.reset_mock()

    @patch("builtins.open", side_effect=lambda path, mode, *args, **kwargs: None)
    def test_generate_services_xml(self, patched_open):
        # Patch file write (to check content) but use original open for reading.
        patched_open.side_effect = self.custom_open

        for number_of_shards, number_of_replicas in self.test_cases:
            with (self.subTest(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas)):
                VespaLocalMultiNode.generate_services_xml(number_of_shards, number_of_replicas)

                # Verify that services.xml is written
                patched_open.assert_any_call('multinode/services.xml', 'w')

                handle = self.write_mock()
                written_xml = "".join([call_arg[0][0] for call_arg in handle.write.call_args_list])

                # Check that written XML exactly matches the expected XML
                expected_file_name = f"expected/services_{number_of_shards}_shard_{number_of_replicas}_replica.xml"
                expected_xml = self._read_file(expected_file_name)
                self.assertEqual(written_xml, expected_xml)

                # Reset call_args_list to avoid tests failing due to previous calls
                self.write_mock.reset_mock()

    @patch("builtins.open", side_effect=lambda path, mode, *args, **kwargs: None)
    def test_generate_hosts_xml(self, patched_open):
        # Patch file write (to check content) but use original open for reading.
        patched_open.side_effect = self.custom_open
        for number_of_shards, number_of_replicas in self.test_cases:
            with self.subTest(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas):
                VespaLocalMultiNode.generate_hosts_xml(number_of_shards, number_of_replicas)

                # Verify that hosts.xml is written
                patched_open.assert_any_call('multinode/hosts.xml', 'w')

                handle = self.write_mock()
                written_xml = "".join([call_arg[0][0] for call_arg in handle.write.call_args_list])

                # Check that written XML exactly matches the expected XML
                expected_file_name = f"expected/hosts_{number_of_shards}_shard_{number_of_replicas}_replica.xml"
                expected_xml = self._read_file(expected_file_name)
                self.assertEqual(written_xml, expected_xml)

                # Reset call_args_list to avoid tests failing due to previous calls
                self.write_mock.reset_mock()
    @patch("os.system")
    @patch("builtins.open")
    def test_start(self, mock_open, mock_system):
        for number_of_shards, number_of_replicas in self.test_cases:
            with self.subTest(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas):
                VespaLocalMultiNode.start(number_of_shards, number_of_replicas)

                # Check that os.system was called to copy and bring up docker compose.
                expected_calls = [
                    call("cp multinode/docker-compose.yml docker-compose.yml"),
                    call("docker compose down 2>/dev/null || true"),
                    call("docker compose up -d"),
                    call("cp multinode/services.xml services.xml"),
                    call("cp multinode/hosts.xml hosts.xml")
                ]
                mock_system.assert_has_calls(expected_calls, any_order=True)


class TestVespaLocalSingleNode(TestVespaLocal):
    @patch("os.system")
    @patch("builtins.open")
    def test_start(self, mock_open, mock_system):
        VespaLocalSingleNode.start()

        # Check that os.system was called to copy and bring up docker compose.
        expected_calls = [
            call("docker rm -f vespa 2>/dev/null || true"),
            call("docker run --detach "
                  "--name vespa "
                  "--hostname vespa-container "
                  "--publish 8080:8080 --publish 19071:19071 --publish 2181:2181 --publish 127.0.0.1:5005:5005 "
                  f"vespaengine/vespa:8.431.32"),
            call("cp singlenode/services.xml services.xml"),
            call("rm -f hosts.xml")
        ]
        mock_system.assert_has_calls(expected_calls, any_order=True)

