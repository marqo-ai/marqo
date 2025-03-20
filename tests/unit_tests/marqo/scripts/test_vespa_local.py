import builtins
import os
import tempfile
from unittest.mock import patch, mock_open, call

from scripts.vespa_local.vespa_local import VespaLocalSingleNode, VespaLocalMultiNode
from tests.unit_tests.marqo_test import MarqoTestCase


class TestVespaLocal(MarqoTestCase):
    def setUp(self):
        # Create a temporary directory and switch to it
        self.test_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir.name)

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
                test_vespa_local = VespaLocalMultiNode(number_of_shards, number_of_replicas)
                test_vespa_local.generate_docker_compose(VESPA_VERSION)

                # Verify that docker-compose.yml is written
                patched_open.assert_any_call('docker-compose.yml', 'w')

                handle = self.write_mock()
                written_yml = "".join([call_arg[0][0] for call_arg in handle.write.call_args_list])

                # Check that written YML exactly matches the expected YML
                expected_file_name = f"expected/docker-compose_{number_of_shards}_shard_{number_of_replicas}_replica.yml"
                expected_yml = self._read_file(expected_file_name)
                self.assertEqual(written_yml, expected_yml)

                # Reset call_args_list to avoid tests failing due to previous calls
                self.write_mock.reset_mock()

    def test_generate_services_xml(self):
        for number_of_shards, number_of_replicas in self.test_cases:
            with (self.subTest(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas)):
                test_vespa_local = VespaLocalMultiNode(number_of_shards, number_of_replicas)
                actual_services_xml_content = test_vespa_local.get_services_xml_content()

                # Check that written XML exactly matches the expected XML
                expected_file_name = f"expected/services_{number_of_shards}_shard_{number_of_replicas}_replica.xml"
                expected_xml = self._read_file(expected_file_name)
                self.assertEqual(actual_services_xml_content, expected_xml)

    def test_generate_hosts_xml(self):
        for number_of_shards, number_of_replicas in self.test_cases:
            with (self.subTest(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas)):
                test_vespa_local = VespaLocalMultiNode(number_of_shards, number_of_replicas)
                actual_hosts_xml_content = test_vespa_local.get_hosts_xml_content()

                # Check that written XML exactly matches the expected XML
                expected_file_name = f"expected/hosts_{number_of_shards}_shard_{number_of_replicas}_replica.xml"
                expected_xml = self._read_file(expected_file_name)
                self.assertEqual(actual_hosts_xml_content, expected_xml)
    @patch("os.system")
    @patch("builtins.open")
    def test_start(self, mock_open, mock_system):
        for number_of_shards, number_of_replicas in self.test_cases:
            with self.subTest(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas):
                test_vespa_local = VespaLocalMultiNode(number_of_shards, number_of_replicas)
                test_vespa_local.start()

                # Check that os.system was called to copy and bring up docker compose.
                expected_calls = [
                    call("docker compose down 2>/dev/null || true"),
                    call("docker compose up -d"),
                ]
                mock_system.assert_has_calls(expected_calls, any_order=True)

    def test_generate_application_package_files_multi_node(self):
        for number_of_shards, number_of_replicas in self.test_cases:
            with self.subTest(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas):
                test_vespa_local = VespaLocalMultiNode(number_of_shards, number_of_replicas)
                test_vespa_local.generate_application_package_files()
                self.assertTrue(os.path.isdir("vespa_dummy_application_package"))
                schema_file = os.path.join("vespa_dummy_application_package", "schemas", "test_vespa_client.sd")
                self.assertTrue(os.path.isfile(schema_file))
                with open(schema_file, "r") as f:
                    content = f.read()
                self.assertIn("schema test_vespa_client", content)
                services_file = os.path.join("vespa_dummy_application_package", "services.xml")
                hosts_file = os.path.join("vespa_dummy_application_package", "hosts.xml")
                self.assertTrue(os.path.isfile(services_file))
                self.assertTrue(os.path.isfile(hosts_file))


class TestVespaLocalSingleNode(TestVespaLocal):
    def setUp(self):
        super().setUp()
        self.test_vespa_local = VespaLocalSingleNode()

    @patch("os.system")
    @patch("builtins.open")
    def test_start(self, mock_open, mock_system):
        self.test_vespa_local.start()

        # Check that os.system was called to copy and bring up docker compose.
        expected_calls = [
            call("docker rm -f vespa 2>/dev/null || true"),
            call("docker run --detach "
                  "--name vespa "
                  "--hostname vespa-container "
                  "--publish 8080:8080 --publish 19071:19071 --publish 2181:2181 --publish 127.0.0.1:5005:5005 "
                  f"vespaengine/vespa:8.472.109"),
        ]
        mock_system.assert_has_calls(expected_calls, any_order=True)

    def test_generate_application_package_files_single_node(self):
        self.test_vespa_local.generate_application_package_files()
        self.assertTrue(os.path.isdir("vespa_dummy_application_package"))
        schema_file = os.path.join("vespa_dummy_application_package", "schemas", "test_vespa_client.sd")
        self.assertTrue(os.path.isfile(schema_file))
        with open(schema_file, "r") as f:
            content = f.read()
        self.assertIn("schema test_vespa_client", content)
        services_file = os.path.join("vespa_dummy_application_package", "services.xml")
        self.assertTrue(os.path.isfile(services_file))