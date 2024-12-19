"""This is a script that pull/start the vespa docker image and deploy a dummy application package.

It can be used in Marqo local runs to start Vespa outside the Marqo docker container. This requires
that the host machine has docker installed.

We generate a schema.sd file and a services.xml file and put them in a zip file. We then deploy the zip file
using the REST API. After that, we check if Vespa is up and running. If it is, we can start Marqo.

All the files are created in a directory called vespa_dummy_application_package. This directory is removed and
the zip file is removed after the application package is deployed.

Note: Vespa CLI is not needed as we use the REST API to deploy the application package.
"""

import os
import shutil
import subprocess
import textwrap
import time
import sys

import requests

VESPA_VERSION=os.getenv('VESPA_VERSION', '8.396.18')  # default version baked into marqo-base:30


def start_vespa() -> None:
    os.system("docker rm -f vespa 2>/dev/null || true")
    os.system("docker run --detach "
              "--name vespa "
              "--hostname vespa-container "
              "--publish 8080:8080 --publish 19071:19071 --publish 2181:2181 "
              f"vespaengine/vespa:{VESPA_VERSION}")


def get_services_xml_content() -> str:
    return textwrap.dedent(
    """<?xml version="1.0" encoding="utf-8" ?>
    <!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
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
            <nodes>
                <node hostalias="node1" distribution-key="0"/>
            </nodes>
        </content>
    </services>
    """)


def get_test_vespa_client_schema_content() -> str:
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


def generate_application_package() -> str:
    base_dir = "vespa_dummy_application_package"
    subdirs = ["schemas"]
    files = {
        "schemas": ["test_vespa_client.sd"],
        "": ["services.xml"]
    }
    # Content for the files
    content_for_services_xml = get_services_xml_content()
    content_for_test_vespa_client_sd = get_test_vespa_client_schema_content()
    # Create the directories and files, and write content
    os.makedirs(base_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        for file in files[subdir]:
            file_path = os.path.join(base_dir, subdir, file)
            with open(file_path, 'w') as f:
                if file == "test_vespa_client.sd":
                    f.write(content_for_test_vespa_client_sd)
    for file in files[""]:
        file_path = os.path.join(base_dir, file)
        with open(file_path, 'w') as f:
            if file == "services.xml":
                f.write(content_for_services_xml)
    os.chdir(base_dir)
    shutil.make_archive('../' + base_dir, 'zip', ".")
    os.chdir("..")
    zip_file_path = f"{base_dir}.zip"

    if os.path.isfile(zip_file_path):
        print(f"Zip file created successfully: {zip_file_path}")
        # Remove the base directory
        shutil.rmtree(base_dir)
        print(f"Directory {base_dir} removed.")
        return zip_file_path
    else:
        print("Failed to create the zip file.")


def deploy_application_package(zip_file_path: str, max_retries: int = 5, backoff_factor: float = 0.5) -> None:
    # URL and headers
    url = "http://localhost:19071/application/v2/tenant/default/prepareandactivate"
    headers = {
        "Content-Type": "application/zip"
    }

    # Ensure the zip file exists
    if not os.path.isfile(zip_file_path):
        print("Zip file does not exist.")
        return

    print("Start deploying the application package...")

    # Attempt to send the request with retries
    for attempt in range(max_retries):
        try:
            with open(zip_file_path, 'rb') as zip_file:
                response = requests.post(url, headers=headers, data=zip_file)
            print(response.text)
            break  # Success, exit the retry loop
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed due to a request error: {e}")
            if attempt < max_retries - 1:
                # Calculate sleep time using exponential backoff
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Aborting.")
                return

    # Cleanup
    os.remove(zip_file_path)
    print("Zip file removed.")


def is_vespa_up(waiting_time: int = 60) -> bool:
    for _ in range(waiting_time):
        document_url = "http://localhost:8080"
        try:
            document_request = requests.get(document_url)
            if document_request.status_code == 200:
                print(f"Vespa is up and running! You can start Marqo. Make sure you set the Vespa environment variable")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)

    print(f"Vespa is not up and running after {waiting_time}s")


def wait_vespa_container_running(max_wait_time: int = 60):
    start_time = time.time()
    # Check if the container is running
    while True:
        if time.time() - start_time > max_wait_time:
            print("Maximum wait time exceeded. Vespa container may not be running.")
            break

        try:
            output = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Status}}", "vespa"])
            if output.decode().strip() == "running":
                print("Vespa container is up and running.")
                break
        except subprocess.CalledProcessError:
            pass

        print("Waiting for Vespa container to start...")
        time.sleep(5)


def main():
    try:
        # Start Vespa
        start_vespa()
        # Wait for the container is pulled and running
        wait_vespa_container_running()
        # Generate the application package
        zip_file_path = generate_application_package()
        # Deploy the application package
        time.sleep(10)
        deploy_application_package(zip_file_path)
        # Check if Vespa is up and running
        is_vespa_up()
    except Exception as e:
        print(f"An error occurred when staring vespa: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


