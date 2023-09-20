from typing import List, Optional, Union, Any, Dict
import subprocess
import os
import json
import requests
from vespa.application import Vespa
from marqo.config import Config
import time

from marqo.vespa.tensor_search.validation import validate_schema_name, validate_application_path


class MarqoVespa:
    """A wrapper for Vespa API calls. This class is used to deploy and manage Vespa applications.
    It calls the Vespa CLI command and contains a Vespa client object.
    """

    def __init__(self, config: Config, index_name: str):
        self.config = config
        self.config_url = os.getenv("VESPA_CONFIG_URL", "http://localhost:19071/")
        self.index_name = index_name
        self.session_id = None

    @staticmethod
    def config_set_target(end_point) -> None:
        command = f"vespa config set target {end_point}"
        subprocess.run(command, shell=True, check=True)

    def check_query_api_ready(self, wait_time: int = 3) -> dict:
        self.config_set_target(self.query_url)
        command = f"vespa status --wait {wait_time}"
        try:
            response = subprocess.run(command, shell=True, check=True, text=True, capture_output=True).stdout
            if "ready" in response:
                return dict({"Status": "ready"})
            else:
                return dict({"Status": "not ready", "message": response})
        except subprocess.CalledProcessError as e:
            return dict({"Status": "error", "message": e.stderr})

    def check_deploy_api_ready(self, wait_time: int = 3) -> dict:
        self.config_set_target(self.config_url)
        command = f"vespa status deploy --wait {wait_time}"
        try:
            response = subprocess.run(command, shell=True, check=True, text=True, capture_output=True).stdout
            if "ready" in response:
                return dict({"Status": "ready"})
            else:
                return dict({"Status": "not ready", "message": response})
        except subprocess.CalledProcessError as e:
            return dict({"Status": "error", "message": e.stderr})

    def deploy(self, application_path: str, wait_time: int = 60) -> str:
        self.config_set_target(self.config_url)
        validated_application_path = validate_application_path(application_path)
        try:
            command = f"vespa deploy --wait {wait_time} {validated_application_path}"
            print("Start Vespa deployment......")
            response = subprocess.run(command, shell=True, check=True, text=True, capture_output=True).stdout
            return response
        except subprocess.CalledProcessError as e:
            print(f'Deployment failed: {e.stderr}')
            return e.stderr

    def start_session(self) -> str:
        # Start a session
        response = json.loads(requests.post(
            f"{self.config_url}/application/v2/tenant/default/session?from="
            f"{self.config_url}/application/v2/tenant/default/application/default/environment"
            f"/default/region/default/instance/default").content)
        self.session_id = response["session-id"]

    def close_session(self) -> None:
        pass

    def get_service_file(self) -> str:
        self.start_session()
        response = self.request_with_retry(
            f"{self.config_url}/application/v2/tenant/default/session/{self.session_id}/content/services.xml")
        return response.content.decode('utf-8')

    def get_hosts_file(self) -> str:
        response = self.request_with_retry(
            f"{self.config_url}/application/v2/tenant/default/session/{self.session_id}/content/hosts.xml")
        if response:
            return response.content.decode('utf-8')
        else:
            return ""

    @staticmethod
    def request_with_retry(url, max_retries=50, retry_delay=0.1):
        """Make a GET request with retries on session NOT_FOUND error.
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 404 and "session not found" in response.text.lower():
                    time.sleep(retry_delay)
                elif response.status_code == 200:
                    return response
            except requests.RequestException as error:
                print(f"An error occurred: {error}")
                break
        else:  # This is executed if the for-loop finishes without a break statement
            print(f"Failed after {max_retries} attempts.")
            return None

    def get_existing_schemas(self) -> List[str]:
        schema_files = []

        # Using the retry function to get the initial list of schema files
        schema_file_list_response = self.request_with_retry(
            f"{self.config_url}/application/v2/tenant/default/session/{self.session_id}/content/schemas/")
        schema_file_list = json.loads(schema_file_list_response.content)

        for schema_file_address in schema_file_list:
            # Using the retry function again to get each individual schema file
            schema_file_response = self.request_with_retry(schema_file_address)
            schema_files.append(schema_file_response.content.decode('utf-8'))
        return schema_files

    def feed_index_settings(self, settings_dict: dict) -> str:
        marqo_settings = [
            {
                "id": self.index_name,  # Unique identifier for the document
                "fields": {
                    "settings": json.dumps(settings_dict)
                }
            },
        ]
        res = self.config.vespa_feed_client.feed_batch(marqo_settings, schema="marqo_settings", asynchronous=False)
        return res