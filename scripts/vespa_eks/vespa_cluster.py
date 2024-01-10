import argparse
import json

import os
import shutil
import re
import subprocess


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary and separates nested keys by sep.

    Args:
    - d (dict): Dictionary to flatten
    - parent_key (str, optional): Key to start with. Defaults to ''.
    - sep (str, optional): Separator to use. Defaults to '.'.

    Returns:
    dict: Flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def replace_values(file_path, replacements):
    """Replace placeholders in a file using a dictionary of replacements."""
    with open(file_path, 'r') as f:
        content = f.read()

    for placeholder, replacement in replacements.items():
        placeholder_pattern = re.escape('{{' + placeholder + '}}')
        content = re.sub(placeholder_pattern, str(replacement), content)

    with open(file_path, 'w') as f:
        f.write(content)


def clone_and_replace(src_dir, dest_dir, config):
    """Clone files from src_dir to dest_dir and replace {{replicas}} with 3."""
    replacements = flatten_dict(config)
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Copy files
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)

        if os.path.isdir(src_path):
            clone_and_replace(src_path, os.path.join(dest_dir, filename), config)
            continue

        dest_path = os.path.join(dest_dir, filename)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
            if filename == 'hosts.xml' or filename == 'services.xml':
                continue  # Don't modify these files

            replace_values(dest_path, replacements)


def create_vespa_hosts(dest_dir, config) -> dict:
    def get_host_node(name, alias):
        return f"""  <host name='vespa-{name}.vespa-internal.default.svc.cluster.local'>
    <alias>{alias}</alias>
  </host>"""

    hosts = dict()

    # Feed container
    feed_replicas = config['feed']['replicas']
    feed_aliases = [f"feed{i}" for i in range(feed_replicas)]
    feed_hosts = '\n'.join([get_host_node(f"feed-container-{i}", f"feed{i}") for i in range(feed_replicas)])
    hosts['feed'] = feed_aliases

    # Query container
    query_replicas = config['query']['replicas']
    query_aliases = [f"query{i}" for i in range(query_replicas)]
    query_hosts = '\n'.join([get_host_node(f"query-container-{i}", f"query{i}") for i in range(query_replicas)])
    hosts['query'] = query_aliases

    # Content cluster
    content_replicas = config['content']['replicas']
    content_aliases = [f"content{i}" for i in range(content_replicas)]
    content_hosts = '\n'.join([get_host_node(f"content-{i}", f"content{i}") for i in range(content_replicas)])
    hosts['content'] = content_aliases

    replace_values(os.path.join(dest_dir, 'hosts.xml'), {"hosts": '\n'.join([feed_hosts, query_hosts, content_hosts])})

    return hosts


def create_vespa_services(dest_dir, hosts: dict):
    feed_nodes = '\n'.join([f"      <node hostalias='{host}' />" for host in hosts['feed']])
    query_nodes = '\n'.join([f"      <node hostalias='{host}' />" for host in hosts['query']])
    content_nodes = '\n'.join([f"      <node hostalias='{host}' distribution-key='{i}' />"
                               for i, host in enumerate(hosts['content'])])

    replace_values(os.path.join(dest_dir, 'services.xml'), {
        "feedNodes": feed_nodes,
        "queryNodes": query_nodes,
        "contentNodes": content_nodes
    })


def prepare(args):
    clean(args)
    config = json.load(open(args.config, 'r'))
    if os.path.exists('prepared'):
        print("Prepared directory already exists. Skipping preparation.")
        return

    here = os.path.dirname(os.path.abspath(__file__))
    clone_and_replace(os.path.join(here, 'template'), 'prepared', config)
    create_vespa_services('prepared', create_vespa_hosts('prepared', config))

    print("Preparation complete.")


def deploy_eks(args):
    prepare(args)

    print(f"Deploying EKS...")
    os.system("eksctl create cluster -f prepared/vespa_cluster.yml")


def deploy_services(args):
    prepare(args)
    kubeconfig(args)

    print(f"Deploying services...")
    os.system("kubectl create -f prepared/config/")


def deploy_config(args):
    prepare(args)
    kubeconfig(args)

    def get_external_ip():
        # Execute the kubectl command and get its output
        result = subprocess.run(['kubectl', 'get', 'svc'], capture_output=True, text=True, check=True)

        # Split the output by lines
        lines = result.stdout.splitlines()

        # Search for the line containing 'vespa-configserver-service'
        for line in lines:
            if 'vespa-configserver-service' in line:
                # Assuming standard kubectl get svc output format, the external IP will be the 4th column.
                columns = line.split()
                return columns[3]

        return None

    config_server = f"http://{get_external_ip()}:19071"
    print(f"Deploying to config server at {config_server}...")
    os.system(f"vespa deploy --target {config_server} prepared")


def clean(args):
    if os.path.exists('prepared'):
        shutil.rmtree('prepared')
        print("Deleted directory 'prepared'.")
    else:
        print("Directory 'prepared' does not exist. Skipping deletion.")


def delete(args):
    prepare(args)
    kubeconfig(args)

    print(args)

    if not args.skip_services:
        print("Deleting services...")
        os.system("kubectl delete -f prepared/config/")
    else:
        print("Skipping deletion of services.")

    print(f"Deleting EKS cluster...")
    os.system("eksctl delete cluster --force -f prepared/vespa_cluster.yml")


def kubeconfig(args):
    config = json.load(open(args.config, 'r'))
    name = config['name']
    region = config['region']

    try:
        result = subprocess.run(['kubectl', 'config', 'current-context'], capture_output=True, text=True, check=True)
        if name == result.stdout.split('/')[-1].strip() and region in result.stdout:
            print("Kubectl config is up to date.")
            return
    except Exception as e:
        print(e)

    os.system(f"aws eks --region {region} update-kubeconfig --name {name}")


def get_endpoints(args):
    kubeconfig(args)

    # Execute the kubectl command and get its output
    result = subprocess.run(['kubectl', 'get', 'svc'], capture_output=True, text=True, check=True)

    # Split the output by lines
    lines = result.stdout.splitlines()

    # Search for the line containing 'vespa-configserver-service'
    for line in lines:
        if 'vespa-configserver-service' in line:
            # Assuming standard kubectl get svc output format, the external IP will be the 4th column.
            columns = line.split()
            endpoint = columns[3]
            port = columns[4].split(':')[0]
            print(f"Config server endpoint: http://{endpoint}:{port}")
        elif 'vespa-feed' in line:
            columns = line.split()
            endpoint = columns[3]
            port = columns[4].split(':')[0]
            print(f"Feed endpoint: http://{endpoint}:{port}")
        elif 'vespa-query' in line:
            columns = line.split()
            endpoint = columns[3]
            port = columns[4].split(':')[0]
            print(f"Query endpoint: http://{endpoint}:{port}")


def main():
    parser = argparse.ArgumentParser(description="CLI for various deployment tasks.")

    subparsers = parser.add_subparsers(title="modes", description="Available modes", help="Deployment modes",
                                       dest='mode')
    subparsers.required = True  # Ensure that a mode is always specified

    # Define the 'prepare' mode with its required 'config' argument
    prepare_parser = subparsers.add_parser("prepare", help="Prepare for deployment")
    prepare_parser.add_argument("config", type=str, help="Path of the config JSON file")
    prepare_parser.set_defaults(func=prepare)

    # Define the 'deploy-eks' mode
    eks_parser = subparsers.add_parser("deploy-eks", help="Deploy EKS")
    eks_parser.add_argument("config", type=str, help="Path of the config JSON file")
    eks_parser.set_defaults(func=deploy_eks)

    # Define the 'deploy-services' mode
    services_parser = subparsers.add_parser("deploy-services", help="Deploy services")
    services_parser.add_argument("config", type=str, help="Path of the config JSON file")
    services_parser.set_defaults(func=deploy_services)

    # Define the 'deploy-config' mode
    config_parser = subparsers.add_parser("deploy-config", help="Deploy config")
    config_parser.add_argument("config", type=str, help="Path of the config JSON file")
    config_parser.set_defaults(func=deploy_config)

    # Define the 'clean' mode
    clean_parser = subparsers.add_parser("clean", help="Clean up resources")
    clean_parser.set_defaults(func=clean)

    # Define the 'delete' mode
    delete_parser = subparsers.add_parser("delete", help="Delete cluster")
    delete_parser.add_argument("config", type=str, help="Path of the config JSON file")
    delete_parser.add_argument("--skip-services", action="store_true", help="Skip deleting services")
    delete_parser.set_defaults(func=delete)

    # Define the 'kubeconfig' mode
    kubeconfig_parser = subparsers.add_parser("kubeconfig", help="Update kubectl cluster")
    kubeconfig_parser.add_argument("config", type=str, help="Path of the config JSON file")
    kubeconfig_parser.set_defaults(func=kubeconfig)

    # Define the 'kubeconfig' mode
    endpoints_parser = subparsers.add_parser("get-endpoints", help="Get Vespa endpoints")
    endpoints_parser.add_argument("config", type=str, help="Path of the config JSON file")
    endpoints_parser.set_defaults(func=get_endpoints)

    # Parse the command-line arguments and execute the corresponding function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no command was provided, print help information
        parser.print_help()


if __name__ == "__main__":
    main()
