class TestVespaLocalMultiNode(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and switch to it
        self.test_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir.name)
        # Ensure multinode directory exists for file writes.
        os.makedirs("multinode", exist_ok=True)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.test_dir.cleanup()

    @patch("builtins.open", new_callable=mock_open)
    def test_generate_docker_compose(self, mock_file):
        number_of_shards = 2
        number_of_replicas = 1  # so total content nodes = 2* (1+1) = 4

        VespaLocalMultiNode.generate_docker_compose(number_of_shards, number_of_replicas, VESPA_VERSION)

        # Ensure docker-compose.yml was written to in multinode directory.
        mock_file.assert_called_with('multinode/docker-compose.yml', 'w')

        # Get the file handle to check its written content.
        handle = mock_file()
        written_data = "".join(call_arg[0][0] for call_arg in handle.write.call_args_list)

        # Load the YAML and test for expected keys
        docker_compose_data = yaml.safe_load(written_data)
        self.assertIn('services', docker_compose_data)
        self.assertIn('networks', docker_compose_data)
        services = docker_compose_data['services']

        # Check that config nodes exist (should be 3)
        for i in range(3):
            self.assertIn(f'config-{i}', services)
            self.assertEqual(services[f'config-{i}']['hostname'], f'config-{i}.vespanet')

        # Check one API node exists (we expect TOTAL_API_NODES = max(MINIMUM_API_NODES, ceil(4/4)) = 1)
        self.assertIn('api-0', services)
        self.assertEqual(services['api-0']['hostname'], 'api-0.vespanet')

        # Check that content nodes exist.
        # With number_of_replicas =1 and shards =2, we expect 2 groups, each with 2 shards.
        self.assertIn('content-0-0', services)
        self.assertIn('content-1-1', services)

    @patch("builtins.open", new_callable=mock_open)
    def test_generate_services_xml(self, mock_file):
        number_of_shards = 2
        number_of_replicas = 1

        VespaLocalMultiNode.generate_services_xml(number_of_shards, number_of_replicas)

        # Verify that services.xml is written
        mock_file.assert_called_with('multinode/services.xml', 'w')
        handle = mock_file()
        written_xml = "".join(call_arg[0][0] for call_arg in handle.write.call_args_list)

        # Parse the XML and verify some expected elements.
        root = ET.fromstring(written_xml)
        self.assertEqual(root.tag, 'services')
        # Check admin/configserver entries (should be 3)
        admin = root.find('admin')
        configservers = admin.find('configservers')
        configserver_list = configservers.findall('configserver')
        self.assertEqual(len(configserver_list), 3)
        # Check that container nodes include the expected API nodes.
        container = root.find('container')
        nodes_elem = container.find('nodes')
        node_list = nodes_elem.findall('node')
        # With 4 total content nodes, TOTAL_API_NODES = max(MINIMUM_API_NODES, ceil(4/4)) = 1
        self.assertGreaterEqual(len(node_list), 1)

        # Check content groups
        content = root.find('content')
        group_parent = content.find('group')
        groups = group_parent.findall('group')
        # number_of_replicas + 1 groups expected
        self.assertEqual(len(groups), number_of_replicas + 1)

    @patch("builtins.open", new_callable=mock_open)
    def test_generate_hosts_xml(self, mock_file):
        number_of_shards = 2
        number_of_replicas = 1

        VespaLocalMultiNode.generate_hosts_xml(number_of_shards, number_of_replicas)

        mock_file.assert_called_with('multinode/hosts.xml', 'w')
        handle = mock_file()
        written_xml = "".join(call_arg[0][0] for call_arg in handle.write.call_args_list)

        root = ET.fromstring(written_xml)
        self.assertEqual(root.tag, 'hosts')

        # Check that 3 config hosts are added
        config_hosts = root.findall("./host[starts-with(@name, 'config')]")
        # Since starts-with is not available, we filter manually.
        config_hosts = [h for h in root.findall("host") if h.attrib['name'].startswith("config")]
        self.assertEqual(len(config_hosts), 3)

        # Check that API nodes are added
        # TOTAL_API_NODES = max(MINIMUM_API_NODES, math.ceil( ( (1+1)*2) / 4)) = 1
        api_hosts = [h for h in root.findall("host") if h.attrib['name'].startswith("api")]
        self.assertEqual(len(api_hosts), 1)

        # Check for content hosts. Expect (number_of_replicas + 1) * number_of_shards = 4 hosts.
        content_hosts = [h for h in root.findall("host") if h.attrib['name'].startswith("content")]
        self.assertEqual(len(content_hosts), (number_of_replicas + 1) * number_of_shards)

    @patch("os.system")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_start(self, mock_file, mock_makedirs, mock_system):
        number_of_shards = 2
        number_of_replicas = 1

        # Ensure multinode directory exists.
        if not os.path.exists("multinode"):
            os.makedirs("multinode")

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