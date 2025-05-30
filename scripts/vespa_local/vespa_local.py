"""This is a script that pull/start the vespa docker image and deploy a dummy application package.
To run everything (single node), use `python vespa_local.py full-start`.
To run everything (multi node), use `python vespa_local.py full-start --Shards 2 --Replicas 1`.

It can be used in Marqo local runs to start Vespa outside the Marqo docker container. This requires
that the host machine has docker installed.

We generate a schema.sd file and a services.xml file and put them in a zip file. We then deploy the zip file
using the REST API. After that, we check if Vespa is up and running. If it is, we can start Marqo.

All the files are created in a directory called vespa_dummy_application_package. This directory is removed and
the zip file is removed after the application package is deployed.

Note: Vespa CLI is not needed for full-start as we use the REST API to deploy the application package.
"""

import os
import shutil
import subprocess
import textwrap
import time
import sys
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
import logging

import requests
import argparse

VESPA_VERSION = os.getenv('VESPA_VERSION', '8.472.109')
VESPA_DISK_USAGE_LIMIT = os.getenv('VESPA_DISK_USAGE_LIMIT', 0.75)
VESPA_CONFIG_URL="http://localhost:19071"
VESPA_DOCUMENT_URL="http://localhost:8080"
VESPA_QUERY_URL="http://localhost:8080"
MINIMUM_API_NODES = 2

# Configure logging: default is INFO. Run script with LogLevel=WARNING to suppress debug logs.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VespaLocal:
    # Base directory for the application package
    base_dir = "vespa_dummy_application_package"
    subdirs = ["schemas"]

    def get_test_vespa_client_schema_content(self) -> str:
        """
        Get the content for the test_vespa_client.sd file. Should be the same for single and multi node vespa.
        """
        return textwrap.dedent("""
            schema test_vespa_client {
                document test_vespa_client {

                    field id type string {
                        indexing: summary | attribute
                    }

                    field title type string {
                        indexing: summary | attribute | index
                        index: enable-bm25
                    }

                    field contents type string {
                        indexing: summary | attribute | index
                        index: enable-bm25
                    }

                }

                fieldset default {
                    fields: title, contents
                }

                rank-profile bm25 inherits default {
                    first-phase {
                        expression: bm25(title) + bm25(contents)
                    }
                }
            }
            """)

    def generate_application_package_files(self):
        """
        Generate files to be zipped for application package
        """

        # Create the directories and files, and write content
        os.makedirs(self.base_dir, exist_ok=True)
        for subdir in self.subdirs:
            os.makedirs(os.path.join(self.base_dir, subdir), exist_ok=True)
            for file in self.application_package_files[subdir]:
                file_path = os.path.join(self.base_dir, subdir, file)
                with open(file_path, 'w') as f:
                    if file == "test_vespa_client.sd":
                        content_for_test_vespa_client_sd = self.get_test_vespa_client_schema_content()
                        f.write(content_for_test_vespa_client_sd)
        for file in self.application_package_files[""]:
            file_path = os.path.join(self.base_dir, file)
            with open(file_path, 'w') as f:
                if file == "services.xml":
                    content_for_services_xml = self.get_services_xml_content()
                    f.write(content_for_services_xml)
                if file == "hosts.xml":
                    # This will only happen for multinode
                    content_for_hosts_xml = self.get_hosts_xml_content()
                    f.write(content_for_hosts_xml)

    def generate_application_package(self) -> str:
        # Build application package directory
        self.generate_application_package_files()

        # Zip up files
        os.chdir(self.base_dir)
        shutil.make_archive('../' + self.base_dir, 'zip', ".")
        os.chdir("..")
        zip_file_path = f"{self.base_dir}.zip"

        if os.path.isfile(zip_file_path):
            logger.info(f"Zip file created successfully: {zip_file_path}")
            # Remove the base directory
            shutil.rmtree(self.base_dir)
            logger.info(f"Directory {self.base_dir} removed.")
            return zip_file_path
        else:
            logger.info("Failed to create the zip file.")
            sys.exit(1)


class VespaLocalSingleNode(VespaLocal):

    def __init__(self):
        self.application_package_files = {
            "schemas": ["test_vespa_client.sd"],
            "": ["services.xml"]
        }
        logger.info("Creating single node Vespa setup.")

    def start(self):
        os.system("docker rm -f vespa 2>/dev/null || true")
        os.system("docker run --detach "
                  "--name vespa "
                  "--hostname vespa-container "
                  "--publish 8080:8080 --publish 19071:19071 --publish 2181:2181 --publish 127.0.0.1:5005:5005 "
                  f"vespaengine/vespa:{VESPA_VERSION}")

    def get_services_xml_content(self) -> str:
        return textwrap.dedent(
            f"""<?xml version="1.0" encoding="utf-8" ?>
            <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
                <container id="default" version="1.0">
                    <document-api/>
                    <search/>
                    <nodes>
                        <node hostalias="node1"/>
                    </nodes>
                </container>
                <content id="content_default" version="1.0">
                    <redundancy>2</redundancy>
                    <documents>
                        <document type="test_vespa_client" mode="index"/>
                    </documents>
                    <tuning>
                        <resource-limits>
                            <disk>{VESPA_DISK_USAGE_LIMIT}</disk>
                        </resource-limits>
                    </tuning>
                    <nodes>
                        <node hostalias="node1" distribution-key="0"/>
                    </nodes>
                </content>
            </services>
            """)

    def wait_vespa_running(self, max_wait_time: int = 60):
        start_time = time.time()
        # Check if the single vespa container is running
        while True:
            if time.time() - start_time > max_wait_time:
                logger.info("Maximum wait time exceeded. Vespa container may not be running.")
                break

            try:
                output = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Status}}", "vespa"])
                if output.decode().strip() == "running":
                    logger.info("Vespa container is up and running.")
                    break
            except subprocess.CalledProcessError:
                pass

            logger.info("Waiting for Vespa container to start...")
            time.sleep(5)


class VespaLocalMultiNode(VespaLocal):

    def __init__(self, number_of_shards, number_of_replicas):
        self.number_of_shards = number_of_shards
        self.number_of_replicas = number_of_replicas
        self.application_package_files = {
            "schemas": ["test_vespa_client.sd"],
            "": ["hosts.xml", "services.xml"]
        }
        logger.info(f"Creating multi-node Vespa setup with {number_of_shards} shards and {number_of_replicas} replicas.")

    def generate_docker_compose(self, vespa_version: str):
        """
        Create docker compose file for multinode vespa with 3 config nodes.
        Generates (number_of_replicas + 1) * number_of shards content nodes.
        """
        services = {}

        logger.info(
            f"Creating `docker-compose.yml` with {self.number_of_shards} shards and {self.number_of_replicas} replicas.")

        BASE_CONFIG_PORT_A = 19071  # configserver (deploy here)
        BASE_SLOBROK_PORT = 19100  # slobrok
        BASE_CLUSTER_CONTROLLER_PORT = 19050  # cluster-controller
        BASE_ZOOKEEPER_PORT = 2181  # zookeeper
        BASE_METRICS_PROXY_PORT = 20092  # metrics-proxy (every node has it)

        BASE_API_PORT_A = 8080  # document/query API
        BASE_DEBUG_PORT = 5005  # debugging port

        BASE_CONTENT_PORT_A = 19107

        TOTAL_CONTENT_NODES = (self.number_of_replicas + 1) * self.number_of_shards
        TOTAL_API_NODES = max(MINIMUM_API_NODES, math.ceil(TOTAL_CONTENT_NODES / 4))
        logger.info(f"Total content nodes: {TOTAL_CONTENT_NODES}, Total API nodes: {TOTAL_API_NODES}")

        # Config Nodes (3)
        nodes_created = 0
        urls_to_health_check = []  # List all API and content node URLs here
        TOTAL_CONFIG_NODES = 3
        for config_node in range(TOTAL_CONFIG_NODES):
            services[f'config-{config_node}'] = {
                'image': f"vespaengine/vespa:{vespa_version or 'latest'}",
                'container_name': f'config-{config_node}',
                'hostname': f'config-{config_node}.vespanet',
                'environment': {
                    'VESPA_CONFIGSERVERS': 'config-0.vespanet,config-1.vespanet,config-2.vespanet',
                    'VESPA_CONFIGSERVER_JVMARGS': '-Xms32M -Xmx128M',
                    'VESPA_CONFIGPROXY_JVMARGS': '-Xms32M -Xmx128M'
                },
                'networks': [
                    'vespanet'
                ],
                'ports': [
                    f'{BASE_CONFIG_PORT_A + config_node}:19071',
                    f'{BASE_SLOBROK_PORT + config_node}:19100',
                    f'{BASE_CLUSTER_CONTROLLER_PORT + config_node}:19050',
                    f'{BASE_ZOOKEEPER_PORT + config_node}:2181',
                    f'{BASE_METRICS_PROXY_PORT + nodes_created}:19092'
                ],
                'command': 'configserver,services',
                'healthcheck': {
                    'test': "curl http://localhost:19071/state/v1/health",
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '40s'
                }
            }
            # Add additional ports to adminserver
            if config_node == 0:
                services[f'config-{config_node}']['ports'].append('19098:19098')

            nodes_created += 1

        # API Nodes
        for api_node in range(TOTAL_API_NODES):
            services[f'api-{api_node}'] = {
                'image': f"vespaengine/vespa:{vespa_version or 'latest'}",
                'container_name': f'api-{api_node}',
                'hostname': f'api-{api_node}.vespanet',
                'environment': [
                    'VESPA_CONFIGSERVERS=config-0.vespanet,config-1.vespanet,config-2.vespanet'
                ],
                'networks': [
                    'vespanet'
                ],
                'ports': [
                    f'{BASE_API_PORT_A + api_node}:8080',
                    f'{BASE_DEBUG_PORT + api_node}:5005',
                    f'{BASE_METRICS_PROXY_PORT + nodes_created}:19092'
                ],
                'command': 'services',
                'depends_on': {
                    'config-0': {'condition': 'service_healthy'},
                    'config-1': {'condition': 'service_healthy'},
                    'config-2': {'condition': 'service_healthy'}
                }
            }
            urls_to_health_check.append(f"http://localhost:{BASE_API_PORT_A + api_node}/state/v1/health")
            nodes_created += 1

        # Content Nodes
        i = 0  # counter of content nodes generated
        for group in range(self.number_of_replicas + 1):
            for shard in range(self.number_of_shards):
                node_name = f'content-{group}-{shard}'
                host_ports = [
                    f'{BASE_CONTENT_PORT_A + i}:19107',
                    f'{BASE_METRICS_PROXY_PORT + nodes_created}:19092'
                ]
                services[node_name] = {
                    'image': f'vespaengine/vespa:{vespa_version or "latest"}',
                    'container_name': node_name,
                    'hostname': f'{node_name}.vespanet',
                    'environment': [
                        'VESPA_CONFIGSERVERS=config-0.vespanet,config-1.vespanet,config-2.vespanet'
                    ],
                    'networks': [
                        'vespanet'
                    ],
                    'ports': host_ports,
                    'command': 'services',
                    'depends_on': {
                        'config-0': {'condition': 'service_healthy'},
                        'config-1': {'condition': 'service_healthy'},
                        'config-2': {'condition': 'service_healthy'}
                    }
                }
                urls_to_health_check.append(f"http://localhost:{BASE_CONTENT_PORT_A + i}/state/v1/health")
                i += 1
                nodes_created += 1

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

        with open('docker-compose.yml', 'w') as f:
            yaml.dump(docker_compose, f, sort_keys=False)
        logger.info(f"Generated `docker-compose.yml` successfully.")

        logger.info("Health check URLs:")
        for url in urls_to_health_check:
            logger.info(url)

    def get_services_xml_content(self):
        """
        Create services.xml for multinode vespa with 3 config nodes.
        Generates (number_of_replicas + 1) groups of number_of shards content nodes each.
        """

        logger.info(f"Writing content for `services.xml` with {self.number_of_shards} shards and {self.number_of_replicas} replicas.")
        TOTAL_CONTENT_NODES = (self.number_of_replicas + 1) * self.number_of_shards
        TOTAL_API_NODES = max(MINIMUM_API_NODES, math.ceil(TOTAL_CONTENT_NODES / 4))
        logger.info(f"Total content nodes: {TOTAL_CONTENT_NODES}, Total API nodes: {TOTAL_API_NODES}")

        # Define the root element with namespaces
        services = ET.Element('services', {
            'version': '1.0',
            'xmlns:deploy': 'vespa',
            'xmlns:preprocess': 'properties'
        })

        # Admin Section
        admin = ET.SubElement(services, 'admin', {'version': '2.0'})

        configservers = ET.SubElement(admin, 'configservers')
        ET.SubElement(configservers, 'configserver', {'hostalias': 'config-0'})
        ET.SubElement(configservers, 'configserver', {'hostalias': 'config-1'})
        ET.SubElement(configservers, 'configserver', {'hostalias': 'config-2'})

        cluster_controllers = ET.SubElement(admin, 'cluster-controllers')
        ET.SubElement(cluster_controllers, 'cluster-controller', {
            'hostalias': 'config-0',
            'jvm-options': '-Xms32M -Xmx64M'
        })
        ET.SubElement(cluster_controllers, 'cluster-controller', {
            'hostalias': 'config-1',
            'jvm-options': '-Xms32M -Xmx64M'
        })
        ET.SubElement(cluster_controllers, 'cluster-controller', {
            'hostalias': 'config-2',
            'jvm-options': '-Xms32M -Xmx64M'
        })

        slobroks = ET.SubElement(admin, 'slobroks')
        ET.SubElement(slobroks, 'slobrok', {'hostalias': 'config-0'})
        ET.SubElement(slobroks, 'slobrok', {'hostalias': 'config-1'})
        ET.SubElement(slobroks, 'slobrok', {'hostalias': 'config-2'})

        # Note: We only have 1 config node for admin.
        ET.SubElement(admin, 'adminserver', {'hostalias': 'config-0'})

        # Container Section (API nodes)
        container = ET.SubElement(services, 'container', {'id': 'default', 'version': '1.0'})
        ET.SubElement(container, 'document-api')
        ET.SubElement(container, 'search')

        nodes = ET.SubElement(container, 'nodes')
        ET.SubElement(nodes, 'jvm', {
            'options': '-Xms32M -Xmx256M -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=*:5005'
        })
        for api_node_number in range(TOTAL_API_NODES):
            ET.SubElement(nodes, 'node', {'hostalias': f'api-{api_node_number}'})

        # Content Section
        content = ET.SubElement(services, 'content', {'id': 'content_default', 'version': '1.0'})
        # Optional: Redundancy can be commented out or adjusted
        redundancy = ET.SubElement(content, 'redundancy')
        redundancy.text = str(self.number_of_replicas + 1)  # As per Vespa's redundancy calculation

        documents = ET.SubElement(content, 'documents')
        ET.SubElement(documents, 'document', {
            'type': 'test_vespa_client',
            'mode': 'index'
        })

        group_parent = ET.SubElement(content, 'group')

        # Distribution configuration
        ET.SubElement(group_parent, 'distribution', {'partitions': '1|' * self.number_of_replicas + "*"})

        # Generate Groups and Nodes
        node_distribution_key = 0
        for group_number in range(self.number_of_replicas + 1):  # +1 for the primary group
            group = ET.SubElement(group_parent, 'group', {
                'name': f'group-{group_number}',
                'distribution-key': str(group_number)
            })
            for shard_number in range(self.number_of_shards):
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

        logger.info("Generated services.xml content successfully!")
        return pretty_xml

    def get_hosts_xml_content(self):
        """
        Create hosts.xml for multinode vespa with 3 config nodes.
        Generates (number_of_replicas + 1) groups of number_of shards content nodes each.
        """

        logger.info(f"Writing content for `hosts.xml` with {self.number_of_shards} shards and {self.number_of_replicas} replicas.")
        TOTAL_CONTENT_NODES = (self.number_of_replicas + 1) * self.number_of_shards
        TOTAL_API_NODES = max(MINIMUM_API_NODES, math.ceil(TOTAL_CONTENT_NODES / 4))
        logger.info(f"Total content nodes: {TOTAL_CONTENT_NODES}, Total API nodes: {TOTAL_API_NODES}")

        # Define the root element
        hosts = ET.Element('hosts')

        # Config Nodes (3)
        config_0 = ET.SubElement(hosts, 'host', {'name': 'config-0.vespanet'})
        alias_config_0 = ET.SubElement(config_0, 'alias')
        alias_config_0.text = 'config-0'

        config_1 = ET.SubElement(hosts, 'host', {'name': 'config-1.vespanet'})
        alias_config_1 = ET.SubElement(config_1, 'alias')
        alias_config_1.text = 'config-1'

        config_2 = ET.SubElement(hosts, 'host', {'name': 'config-2.vespanet'})
        alias_config_2 = ET.SubElement(config_2, 'alias')
        alias_config_2.text = 'config-2'

        # API Nodes (container)
        for api_node_number in range(TOTAL_API_NODES):
            api_node = ET.SubElement(hosts, 'host',
                                     {'name': f'api-{api_node_number}.vespanet'})
            alias_api_node = ET.SubElement(api_node, 'alias')
            alias_api_node.text = f'api-{api_node_number}'

        # Content Nodes
        for group_number in range(self.number_of_replicas + 1):  # +1 for the primary group
            for shard_number in range(self.number_of_shards):
                content_node = ET.SubElement(hosts, 'host',
                                             {'name': f'content-{group_number}-{shard_number}.vespanet'})
                alias_content_node = ET.SubElement(content_node, 'alias')
                alias_content_node.text = f'content-{group_number}-{shard_number}'

        # Convert the ElementTree to a string
        rough_string = ET.tostring(hosts, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml_bytes = reparsed.toprettyxml(indent="    ", encoding='utf-8')
        pretty_xml = pretty_xml_bytes.decode('utf-8')

        logger.info("Generated hosts.xml content successfully!")
        return pretty_xml

    def start(self):
        # Generate the docker compose file
        self.generate_docker_compose(
            vespa_version=VESPA_VERSION
        )

        # Start the docker compose
        os.system("docker compose down 2>/dev/null || true")
        os.system("docker compose up -d")

    def wait_vespa_running(self, max_wait_time: int = 20):
        # Just wait 20 seconds
        logger.info(f"Waiting for Vespa to start for {max_wait_time} seconds.")
        time.sleep(max_wait_time)


def container_exists(container_name):
    import docker   # Only try importing docker here. Not needed for other functions.
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        return True
    except docker.errors.NotFound:
        return False
    except docker.errors.APIError as e:
        logger.info(f"Error accessing Docker API: {e}")
        return False


def validate_shards_and_replicas_count(args):
    if args.Shards < 1:
        raise ValueError("Number of shards must be at least 1.")
    if args.Replicas < 0:
        raise ValueError("Number of replicas must be at least 0.")


# Callable functions from workflows or CLI
# These functions will call the appropriate methods from single/multi node vespa setups.
def full_start(args):
    # Create instance of VespaLocal
    # vespa_local_instance is used for starting vespa & generating application package.
    validate_shards_and_replicas_count(args)
    if args.Shards > 1 or args.Replicas > 0:
        vespa_local_instance = VespaLocalMultiNode(args.Shards, args.Replicas)
    else:
        vespa_local_instance = VespaLocalSingleNode()

    # Start Vespa
    vespa_local_instance.start()
    # Wait until vespa is up and running
    vespa_local_instance.wait_vespa_running()
    # Generate the application package
    zip_file_path = vespa_local_instance.generate_application_package()
    # Deploy the application package
    time.sleep(10)
    deploy_application_package(zip_file_path)
    # Check if Vespa is up and running
    has_vespa_converged()


def start(args):
    # Normal start command without deploying (recommended to use full_start instead)
    # Create instance of VespaLocal
    # vespa_local_instance is used for starting vespa & generating application package.
    validate_shards_and_replicas_count(args)
    if args.Shards > 1 or args.Replicas > 0:
        vespa_local_instance = VespaLocalMultiNode(args.Shards, args.Replicas)
    else:
        vespa_local_instance = VespaLocalSingleNode()

    vespa_local_instance.start()


def restart(args):
    if container_exists("vespa"):
        logger.info("Single Node Vespa setup found (container with name 'vespa'). Restarting container.")
        os.system("docker restart vespa")
    else:
        logger.info("Assuming Multi Node Vespa setup. Restarting all containers.")
        os.system("docker compose restart")


def stop(args):
    if container_exists("vespa"):
        logger.info("Single Node Vespa setup found (container with name 'vespa'). Stopping container.")
        os.system("docker stop vespa")
    else:
        logger.info("Assuming Multi Node Vespa setup. Stopping and removing all containers.")
        os.system("docker compose down")


def deploy_config(args):
    """
    Deploy the config using Vespa CLI assuming this directory contains the vespa application files
    """
    os.system('vespa config set target local')
    here = os.path.dirname(os.path.abspath(__file__))
    os.system(f'vespa deploy "{here}"')



def deploy_application_package(zip_file_path: str, max_retries: int = 5, backoff_factor: float = 0.5) -> None:
    # URL and headers
    url = f"{VESPA_CONFIG_URL}/application/v2/tenant/default/prepareandactivate"
    headers = {
        "Content-Type": "application/zip"
    }

    # Ensure the zip file exists
    if not os.path.isfile(zip_file_path):
        logger.info("Zip file does not exist.")
        return

    logger.info("Start deploying the application package...")

    # Attempt to send the request with retries
    for attempt in range(max_retries):
        try:
            with open(zip_file_path, 'rb') as zip_file:
                response = requests.post(url, headers=headers, data=zip_file)
            logger.info(response.text)
            break  # Success, exit the retry loop
        except requests.exceptions.RequestException as e:
            logger.info(f"Attempt {attempt + 1} failed due to a request error: {e}")
            if attempt < max_retries - 1:
                # Calculate sleep time using exponential backoff
                sleep_time = backoff_factor * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.info("Max retries reached. Aborting.")
                return

    # Cleanup
    os.remove(zip_file_path)
    logger.info("Zip file removed.")


def generate_and_deploy_application_package(args):
    # For print messages for cleaner logs (call this function with --LogLevel INFO)
    numeric_level = getattr(logging, args.LogLevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    logger.setLevel(numeric_level)

    # Create instance of VespaLocal
    # vespa_local_instance is used for starting vespa & generating application package.
    if args.Shards > 1 or args.Replicas > 0:
        vespa_local_instance = VespaLocalMultiNode(args.Shards, args.Replicas)
    else:
        vespa_local_instance = VespaLocalSingleNode()
    # Generate the application package
    zip_file_path = vespa_local_instance.generate_application_package()
    # Deploy the application package
    deploy_application_package(zip_file_path)


def has_vespa_converged(waiting_time: int = 600) -> bool:
    logger.info("Checking if Vespa has converged...")
    converged = False
    start_time = time.time()
    while time.time() - start_time < waiting_time:
        try:
            response = requests.get(
                f"{VESPA_CONFIG_URL}/application/v2/tenant/default/application/default/environment/prod/region/"
                f"default/instance/default/serviceconverge")
            data = response.json()
            if data.get('converged') == True:
                converged = True
                break
            logger.info("  Waiting for Vespa convergence to be true...")
        except Exception as e:
            logger.info(f"  Error checking convergence: {str(e)}")

        time.sleep(10)

    if not converged:
        logger.info("Vespa did not converge in time")
        sys.exit(1)

    logger.info("Vespa application has converged. Vespa setup complete!")


def main():
    parser = argparse.ArgumentParser(description="CLI for local Vespa deployment.")
    subparsers = parser.add_subparsers(title="modes", description="Available modes", help="Deployment modes",
                                       dest='mode')
    subparsers.required = True  # Ensure that a mode is always specified

    full_start_parser = subparsers.add_parser("full-start",
                                         help="Start local Vespa, build package, deploy, and wait for readiness.")
    full_start_parser.set_defaults(func=full_start)
    full_start_parser.add_argument('--Shards', help='The number of shards', default=1, type=int)
    full_start_parser.add_argument('--Replicas', help='The number of replicas', default=0, type=int)

    start_parser = subparsers.add_parser("start",
                                         help="Start local Vespa only")
    start_parser.set_defaults(func=start)
    start_parser.add_argument('--Shards', help='The number of shards', default=1, type=int)
    start_parser.add_argument('--Replicas', help='The number of replicas', default=0, type=int)

    prepare_parser = subparsers.add_parser("restart", help="Restart existing local Vespa")
    prepare_parser.set_defaults(func=restart)

    eks_parser = subparsers.add_parser("deploy-config", help="Deploy config")
    eks_parser.set_defaults(func=deploy_config)     # TODO: Set this to deploy_application_package

    clean_parser = subparsers.add_parser("stop", help="Stop local Vespa")
    clean_parser.set_defaults(func=stop)

    # This function is used in prod (run_marqo.sh)
    generate_and_deploy_parser = subparsers.add_parser("generate-and-deploy", help="Generate and deploy application package")
    generate_and_deploy_parser.set_defaults(func=generate_and_deploy_application_package)
    generate_and_deploy_parser.add_argument('--Shards', help='The number of shards', default=1, type=int)
    generate_and_deploy_parser.add_argument('--Replicas', help='The number of replicas', default=0, type=int)
    generate_and_deploy_parser.add_argument('--LogLevel',
                                            help='Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)',
                                            default='INFO', type=str)


    # Parse the command-line arguments and execute the corresponding function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no command was provided, print help information
        parser.print_help()


if __name__ == "__main__":
    main()


