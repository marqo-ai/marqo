import argparse

import os
import yaml
import docker
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

VESPA_VERSION=os.getenv('VESPA_VERSION', '8.431.32')  # default version baked into marqo-base:44

class VespaLocalSingleNode:
    def __init__(self):
        pass

    @classmethod
    def start(cls):
        os.system("docker rm -f vespa 2>/dev/null || true")
        os.system("docker run --detach "
                  "--name vespa "
                  "--hostname vespa-container "
                  "--publish 8080:8080 --publish 19071:19071 --publish 2181:2181 --publish 127.0.0.1:5005:5005 "
                  f"vespaengine/vespa:{VESPA_VERSION}")

        # Copy services.xml to base dir. It can be zipped from here.
        os.system("cp singlenode/services.xml services.xml")
        print("Copied services.xml to base directory.")


class VespaLocalMultiNode:
    def __init__(self):
        pass

    @classmethod
    def generate_docker_compose(cls, number_of_shards: int, number_of_replicas: int, vespa_version: str):
        """
        Create docker compose file for multinode vespa with 3 config nodes.
        Generates (number_of_replicas + 1) * number_of shards content nodes.
        """
        services = {}

        # TODO: Find better name for this
        BASE_CONTENT_PORT_A = 19107
        BASE_CONTENT_PORT_B = 20100

        # TODO: change this to 3 config nodes.
        # Config Node (config-0)
        services['config-0'] = {
            'image': f"vespaengine/vespa:{vespa_version or 'latest'}",
            'container_name': 'config-0',
            'hostname': 'config-0.vespanet',
            'environment': [
                'VESPA_CONFIGSERVERS=config-0.vespanet'
            ],
            'networks': [
                'vespanet'
            ],
            'ports': [
                '19071:19071',
                '8080:8080',
                '5005:5005'
            ],
            'command': 'configserver,services'
        }

        # Generate Content Nodes
        i = 0  # counter of content nodes generated
        for group in range(number_of_replicas + 1):
            for shard in range(number_of_shards):
                node_name = f'content-{group}-{shard}'
                host_ports = [
                    f'{BASE_CONTENT_PORT_A + i}:19107',
                    f'{BASE_CONTENT_PORT_B + i}:19092'
                ]
                services[node_name] = {
                    'image': f'vespaengine/vespa:{vespa_version or "latest"}',
                    'container_name': node_name,
                    'hostname': f'{node_name}.vespanet',
                    'environment': [
                        'VESPA_CONFIGSERVERS=config-0.vespanet'
                    ],
                    'networks': [
                        'vespanet'
                    ],
                    'ports': host_ports,
                    'command': 'services'
                }
                i += 1

        # Define Networks
        networks = {
            'vespanet': {
                'driver': 'bridge'
            }
        }

        # Combine into final docker-compose structure
        docker_compose = {
            'services': services,
            'networks': networks
        }

        with open('multinode/docker-compose.yml', 'w') as f:
            yaml.dump(docker_compose, f, sort_keys=False)
        print(f"Generated `multinode/docker-compose.yml` successfully.")

    @classmethod
    def generate_services_xml(cls, number_of_shards: int, number_of_replicas: int):
        """
        Create services.xml for multinode vespa with 3 config nodes.
        Generates (number_of_replicas + 1) groups of number_of shards content nodes each.
        """

        print(f"Creating `multinode/services.xml` with {number_of_shards} shards and {number_of_replicas} replicas.")

        # Define the root element with namespaces
        services = ET.Element('services', {
            'version': '1.0',
            'xmlns:deploy': 'vespa',
            'xmlns:preprocess': 'properties'
        })

        # Admin Section
        # TODO: Change to 3 config servers
        admin = ET.SubElement(services, 'admin', {'version': '2.0'})

        configservers = ET.SubElement(admin, 'configservers')
        ET.SubElement(configservers, 'configserver', {'hostalias': 'config-0'})

        cluster_controllers = ET.SubElement(admin, 'cluster-controllers')
        ET.SubElement(cluster_controllers, 'cluster-controller', {
            'hostalias': 'config-0',
            'jvm-options': '-Xms32M -Xmx64M'
        })

        slobroks = ET.SubElement(admin, 'slobroks')
        ET.SubElement(slobroks, 'slobrok', {'hostalias': 'config-0'})

        ET.SubElement(admin, 'adminserver', {'hostalias': 'config-0'})

        # Container Section
        container = ET.SubElement(services, 'container', {'id': 'default', 'version': '1.0'})
        ET.SubElement(container, 'document-api')
        ET.SubElement(container, 'search')

        nodes = ET.SubElement(container, 'nodes')
        ET.SubElement(nodes, 'jvm', {
            'options': '-Xms32M -Xmx256M -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=*:5005'
        })
        ET.SubElement(nodes, 'node', {'hostalias': 'config-0'})

        # Content Section
        content = ET.SubElement(services, 'content', {'id': 'content_default', 'version': '1.0'})
        # Optional: Redundancy can be commented out or adjusted
        redundancy = ET.SubElement(content, 'redundancy')
        redundancy.text = str(number_of_replicas + 1)  # As per Vespa's redundancy calculation

        documents = ET.SubElement(content, 'documents')
        ET.SubElement(documents, 'document', {
            'type': 'test_vespa_client',
            'mode': 'index'
        })

        group_parent = ET.SubElement(content, 'group')

        # Distribution configuration
        ET.SubElement(group_parent, 'distribution', {'partitions': '1|' * number_of_replicas + "*"})

        # Generate Groups and Nodes
        node_distribution_key = 0
        for group_number in range(number_of_replicas + 1): # +1 for the primary group
            group = ET.SubElement(group_parent, 'group', {
                'name': f'group-{group_number}',
                'distribution-key': str(group_number)
            })
            for shard_number in range(number_of_shards):
                hostalias = f'content-{group_number}-{shard_number}'
                ET.SubElement(group, 'node', {
                    'hostalias': hostalias,
                    'distribution-key': str(node_distribution_key)
                })
                node_distribution_key += 1

        # Convert the ElementTree to a string
        rough_string = ET.tostring(services, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml_bytes = reparsed.toprettyxml(indent="    ", encoding='utf-8')
        pretty_xml = pretty_xml_bytes.decode('utf-8')

        # Write to the output file
        with open('multinode/services.xml', 'w') as f:
            f.write(pretty_xml)

        print(f"Generated multinode/services.xml successfully.")

    @classmethod
    def generate_hosts_xml(cls, number_of_shards: int, number_of_replicas: int):
        """
        Create hosts.xml for multinode vespa with 3 config nodes.
        Generates (number_of_replicas + 1) groups of number_of shards content nodes each.
        """

        print(f"Creating `multinode/hosts.xml` with {number_of_shards} shards and {number_of_replicas} replicas.")

        # Define the root element
        hosts = ET.Element('hosts')

        # Config Nodes
        # TODO: Change to 3 config servers
        config_0 = ET.SubElement(hosts, 'host', {'name': 'config-0.vespanet'})
        alias_config_0 = ET.SubElement(config_0, 'alias')
        alias_config_0.text = 'config-0'

        # Content Nodes
        for group_number in range(number_of_replicas + 1):  # +1 for the primary group
            for shard_number in range(number_of_shards):
                content_node = ET.SubElement(hosts, 'host',
                                             {'name': f'content-{group_number}-{shard_number}.vespanet'})
                alias_content_node = ET.SubElement(content_node, 'alias')
                alias_content_node.text = f'content-{group_number}-{shard_number}'

        # Convert the ElementTree to a string
        rough_string = ET.tostring(hosts, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml_bytes = reparsed.toprettyxml(indent="    ", encoding='utf-8')
        pretty_xml = pretty_xml_bytes.decode('utf-8')

        # Write to the output file
        with open('multinode/hosts.xml', 'w') as f:
            f.write(pretty_xml)

        print(f"Generated multinode/hosts.xml successfully.")

    @classmethod
    def start(cls, number_of_shards, number_of_replicas):
        if not os.path.exists("multinode"):
            os.makedirs("multinode")

        # Generate the docker compose file
        VespaLocalMultiNode.generate_docker_compose(
            number_of_shards=number_of_shards,
            number_of_replicas=number_of_replicas,
            vespa_version=VESPA_VERSION
        )

        # Start the docker compose
        os.system("cp multinode/docker-compose.yml docker-compose.yml")
        os.system("docker compose down 2>/dev/null || true")
        os.system("docker compose up -d")

        # Generate the services.xml and hosts.xml
        hosts_xml = VespaLocalMultiNode.generate_hosts_xml(
            number_of_shards=number_of_shards,
            number_of_replicas=number_of_replicas
        )

        services_xml = VespaLocalMultiNode.generate_services_xml(
            number_of_shards=number_of_shards,
            number_of_replicas=number_of_replicas
        )

        # Copy services.xml and hosts.xml to base dir. They can be zipped from here.
        os.system("cp multinode/services.xml services.xml")
        os.system("cp multinode/hosts.xml hosts.xml")
        print("Copied services.xml and hosts.xml to base directory.")

    @classmethod
    def deploy_config(cls):
        os.system('vespa config set target local')
        here = os.path.dirname(os.path.abspath(__file__))
        os.system(f'vespa deploy "{here}"')


def container_exists(container_name):
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        return True
    except docker.errors.NotFound:
        return False
    except docker.errors.APIError as e:
        print(f"Error accessing Docker API: {e}")
        return False


# Callable functions from workflows or CLI
# These functions will call the appropriate methods from single/multi node vespa setups.
def start(args):
    if args.Shards > 1 or args.Replicas > 0:
        VespaLocalMultiNode.start(args.Shards, args.Replicas)
    else:
        VespaLocalSingleNode.start()

def restart(args):
    if container_exists("vespa"):
        print("Single Node Vespa setup found (container with name 'vespa'). Restarting container.")
        os.system("docker restart vespa")
    else:
        print("Assuming Multi Node Vespa setup. Restarting all containers.")
        os.system("docker compose restart")

def stop(args):
    if container_exists("vespa"):
        print("Single Node Vespa setup found (container with name 'vespa'). Stopping container.")
        os.system("docker stop vespa")
    else:
        print("Assuming Multi Node Vespa setup. Stopping and removing all containers.")
        os.system("docker compose down")


def deploy_config(args):
    os.system('vespa config set target local')
    here = os.path.dirname(os.path.abspath(__file__))
    os.system(f'vespa deploy "{here}"')


def main():
    parser = argparse.ArgumentParser(description="CLI for local Vespa deployment.")
    subparsers = parser.add_subparsers(title="modes", description="Available modes", help="Deployment modes",
                                       dest='mode')
    subparsers.required = True  # Ensure that a mode is always specified

    start_parser = subparsers.add_parser("start", help="Start local Vespa")
    start_parser.set_defaults(func=start)
    start_parser.add_argument('--Shards', help='The number of shards', default=1, type=int)
    start_parser.add_argument('--Replicas', help='The number of replicas', default=0, type=int)

    prepare_parser = subparsers.add_parser("restart", help="Restart existing local Vespa")
    prepare_parser.set_defaults(func=restart)

    eks_parser = subparsers.add_parser("deploy-config", help="Deploy config")
    eks_parser.set_defaults(func=deploy_config)

    clean_parser = subparsers.add_parser("stop", help="Stop local Vespa")
    clean_parser.set_defaults(func=stop)

    # Parse the command-line arguments and execute the corresponding function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no command was provided, print help information
        parser.print_help()


if __name__ == "__main__":
    main()
