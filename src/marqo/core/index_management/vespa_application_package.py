import json
import os
import io
import tarfile
import tempfile
import textwrap
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Tuple, Generator, Dict

import semver
from datetime import datetime

from pathlib import Path
import xml.etree.ElementTree as ET

from marqo.base_model import ImmutableBaseModel
from marqo.core.exceptions import InternalError, OperationConflictError, IndexNotFoundError, IndexExistsError, \
    ApplicationRollbackError
from marqo.core.models import MarqoIndex
import marqo.logging
from marqo.vespa.exceptions import VespaError
from marqo.vespa.vespa_client import VespaClient


logger = marqo.logging.get_logger(__name__)


class ServicesXml:
    """
    Represents a Vespa services XML file.
    """
    _CUSTOM_COMPONENT_BUNDLE_NAME = "marqo-custom-searchers"

    def __init__(self, xml_str: str):
        self._root = ET.fromstring(xml_str)
        self._documents = self._ensure_only_one('content/documents')

    def __repr__(self) -> str:
        return f'ServicesXml({self.to_xml()})'

    def to_xml(self) -> str:
        # Add the namespace attribute to the root element
        self._root.set('xmlns:deploy', 'vespa')
        self._root.set('xmlns:preprocess', 'properties')

        xml_declaration = '<?xml version="1.0" encoding="utf-8" ?>\n'
        return xml_declaration + ET.tostring(self._root).decode('utf-8')

    def _ensure_only_one(self, xml_path: str) -> ET.Element:
        elements = self._root.findall(xml_path)

        if len(elements) > 1:
            raise InternalError(f'Multiple {xml_path} elements found in services.xml. Only one is allowed')
        if len(elements) == 0:
            raise InternalError(f'No {xml_path} element found in services.xml')

        return elements[0]

    def add_schema(self, name: str) -> None:
        if self._documents.find(f'document[@type="{name}"]') is not None:
            logger.warn(f'Schema {name} already exists in services.xml, nothing to add')
        else:
            new_document = ET.SubElement(self._documents, 'document')
            new_document.set('type', name)
            new_document.set('mode', 'index')

    def remove_schema(self, name: str) -> None:
        docs = self._documents.findall(f'document[@type="{name}"]')
        if not docs:
            logger.warn(f'Schema {name} does not exist in services.xml, nothing to remove')
        else:
            for doc in docs:
                self._documents.remove(doc)

    def config_components(self):
        self._cleanup_container_config()
        self._config_search()
        self._config_index_setting_components()

    def _cleanup_container_config(self):
        """
        Components config needs to be in sync with the components in the jar files. This method cleans up the
        custom components config, so we can always start fresh. This assumes that the container section of the
        services.xml file only has `node` config and empty `document-api` and `search` elements initially. Please
        note that any manual config change in container section will be overwritten.
        """
        container_element = self._ensure_only_one('container')
        for child in container_element.findall('*'):
            if child.tag in ['document-api', 'search']:
                child.clear()
            elif child.tag != 'nodes':
                container_element.remove(child)

    def _config_search(self):
        search_elements = self._ensure_only_one('container/search')
        chain = ET.SubElement(search_elements, 'chain')
        chain.set('id', 'marqo')
        chain.set('inherits', 'vespa')
        self._add_component(chain, 'searcher', 'ai.marqo.search.HybridSearcher')
        
    def _config_index_setting_components(self):
        container_elements = self._ensure_only_one('container')

        # Add index setting handler
        index_setting_handler = self._add_component(container_elements, 'handler',
                                                    'ai.marqo.index.IndexSettingRequestHandler')
        for binding in ['http://*/index-settings/*', 'http://*/index-settings']:
            binding_element = ET.SubElement(index_setting_handler, 'binding')
            binding_element.text = binding

        # Add index setting component
        index_setting_component = self._add_component(container_elements, 'component',
                                                      'ai.marqo.index.IndexSettings')
        config_element = ET.SubElement(index_setting_component, 'config')
        config_element.set('name', 'ai.marqo.index.index-settings')
        ET.SubElement(config_element, 'indexSettingsFile').text = 'marqo_index_settings.json'
        ET.SubElement(config_element, 'indexSettingsHistoryFile').text = 'marqo_index_settings_history.json'

    def _add_component(self, parent: ET.Element, tag: str, name: str):
        element = ET.SubElement(parent, tag)
        element.set('id', name)
        element.set('bundle', self._CUSTOM_COMPONENT_BUNDLE_NAME)
        return element

    def compare_element(self, other: 'ServicesXml', xml_path: str) -> bool:
        def normalize(elem: ET.Element):
            # Sort attributes and child elements to normalize
            normalized = ET.Element(elem.tag, dict(sorted(elem.attrib.items())))
            children = [normalize(child) for child in elem.findall('*')]
            for child in sorted(children, key=lambda x: str((x.tag, x.attrib))):
                normalized.append(child)
            return normalized

        elements_self = sorted([ET.tostring(normalize(elem), encoding='unicode') for elem in self._root.findall(xml_path)])
        elements_other = sorted([ET.tostring(normalize(elem), encoding='unicode') for elem in other._root.findall(xml_path)])

        return len(elements_self) == len(elements_other) and all(x == y for x, y in zip(elements_self, elements_other))


class IndexSettingStore:
    """
    Index settings are now stored in a json file within the application package. Up to 3 historical versions of each
    index setting are stored in another json file. This class handles the creation and management of index settings

    Index settings json file format:
    {
        "index1": {
            "name": "index1",
            "version": 1,
            ...
        }
    }

    Index settings history json file format:
    {
        "index2": [
            {
                "name": "index2",
                "version": 2,
               ...
            },
            {
                "name": "index2",
                "version": 1,
               ...
            }
        ]
    }
    """
    _HISTORY_VERSION_LIMIT = 3

    def __init__(self, index_settings_json: str, index_settings_history_json: str):
        self._index_settings: Dict[str, MarqoIndex] = \
            {key: MarqoIndex.parse_obj(value) for key, value in json.loads(index_settings_json).items()}

        self._index_settings_history: Dict[str, List[MarqoIndex]] = \
            {key: [MarqoIndex.parse_obj(value) for value in history_array]
             for key, history_array in json.loads(index_settings_history_json).items()}

    def to_json(self) -> (str, str):
        # Pydantic 1.x does not support creating jsonable dict. `json.loads(index.json())` is a workaround
        json_str = json.dumps({key: json.loads(index.json()) for key, index in self._index_settings.items()})
        history_str = json.dumps({key: [json.loads(index.json()) for index in history] for
                                  key, history in self._index_settings_history.items()})

        return json_str, history_str

    def save_index_setting(self, index_setting: MarqoIndex) -> None:
        target_version = index_setting.version or 1
        name = index_setting.name

        if name in self._index_settings:
            current_version = self._index_settings[name].version
            if current_version + 1 != target_version:
                raise OperationConflictError(f'Conflict in version detected while saving index {name}. '
                                             f'Current version is {current_version}, and cannot be upgraded to '
                                             f'target version {target_version}. Some other request might have changed '
                                             f'the index. Please try again. ')
            self._move_to_history(name)
        else:
            if target_version != 1:
                raise OperationConflictError(f'Conflict in version detected while saving index {name}. '
                                             f'The index does not exist or has been deleted, and we are trying to '
                                             f'upgrade it to version {target_version}')
            # If the name exists in history, it means index with the same name was deleted. Clean the history
            if name in self._index_settings_history:
                del self._index_settings_history[name]

        self._index_settings[name] = index_setting.copy(deep=True, update={'version': target_version})

    def delete_index_setting(self, index_setting_name: str) -> None:
        if index_setting_name not in self._index_settings:
            logger.warn(f"Index setting {index_setting_name} does not exist, nothing to delete")
        else:
            self._move_to_history(index_setting_name)
            del self._index_settings[index_setting_name]

    def get_index(self, index_setting_name: str) -> MarqoIndex:
        return self._index_settings[index_setting_name] if index_setting_name in self._index_settings else None

    def _move_to_history(self, index_setting_name):
        if index_setting_name in self._index_settings_history:
            self._index_settings_history[index_setting_name].insert(0, self._index_settings[index_setting_name])
            self._index_settings_history[index_setting_name] = self._index_settings_history[index_setting_name][
                                                               :self._HISTORY_VERSION_LIMIT]
        else:
            self._index_settings_history[index_setting_name] = [self._index_settings[index_setting_name]]


class MarqoConfig(ImmutableBaseModel):
    version: str

    class Config(ImmutableBaseModel.Config):
        extra = "allow"


class MarqoConfigStore:
    """
    Store Marqo configuration in a JSON file in the application package
    Index settings json file format:
    {
        "version": "2.11.0"
        ...
    }
    """
    def __init__(self, marqo_config_json: str) -> None:
        self._config = MarqoConfig.parse_obj(json.loads(marqo_config_json)) if marqo_config_json else None

    def get(self) -> MarqoConfig:
        return self._config

    def update_version(self, version: str) -> None:
        if self._config is None:
            self._config = MarqoConfig(version=version)
        else:
            self._config = self._config.copy(update={'version': version})


class VespaAppBackup:
    """
    This class represents a backup zip file for a Vespa application package. It contains all the config files we changed
    or added in the previous bootstrapping process. (Binary files are excluded.) During a rollback, we will recover all
    changed files, and remove all added files from this zip file.
    """
    _REMOVE_FILE_LIST = "files_to_remove.json"

    def __init__(self, backup_zip_file_content: Optional[bytes] = None) -> None:
        self._dir = tempfile.mkdtemp()
        self._removal_mark_file = os.path.join(self._dir, self._REMOVE_FILE_LIST)
        self._files_to_remove = []
        self._files_to_rollback = []

        if backup_zip_file_content is not None:
            self._extract_gzip_from_bytes(backup_zip_file_content)
            if os.path.isfile(self._removal_mark_file):
                self._files_to_remove = json.load(open(self._removal_mark_file))
                os.remove(self._removal_mark_file)

    def __repr__(self):
        return (f'<VespaAppBackup files_to_rollback={[os.path.join(*paths) for paths in self._files_to_rollback]} '
                f'files_to_remove={[os.path.join(*paths) for paths in self._files_to_remove]}>')

    def read_text_file(self, *paths: str) -> Optional[str]:
        path = os.path.join(self._dir, *paths)
        if not os.path.isfile(path):
            return None
        with open(path, 'r') as f:
            return f.read()

    def files_to_rollback(self) -> Generator[Tuple[Tuple[str, ...], bytes], None, None]:
        for root, dirs, files in os.walk(self._dir):
            for file in files:
                file_path = os.path.join(root, file)
                relpath = os.path.relpath(file_path, self._dir)
                with open(file_path, "rb") as f:
                    # TODO verify if this works on windows
                    yield relpath, f.read()

    def files_to_remove(self) -> Generator[Tuple[str, ...], None, None]:
        for paths in self._files_to_remove:
            yield paths

    def backup_file(self, content: Union[str, bytes], *paths: str) -> None:
        path = os.path.join(self._dir, *paths)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        mode = 'w' if isinstance(content, str) else 'wb'
        with open(path, mode) as f:
            f.write(content)

        self._files_to_rollback.append(paths)

    def mark_for_removal(self, *paths: str) -> None:
        self._files_to_remove.append(paths)

    def to_zip_stream(self) -> io.BytesIO:
        with open(self._removal_mark_file, 'w') as f:
            f.write(json.dumps(self._files_to_remove))

        byte_stream = io.BytesIO()
        with tarfile.open(fileobj=byte_stream, mode='w:gz') as tar:
            for root, dirs, files in os.walk(self._dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_name = os.path.relpath(file_path, self._dir)  # archive name should be relative
                    tar.add(file_path, arcname=archive_name)

        byte_stream.seek(0)
        return byte_stream

    def _extract_gzip_from_bytes(self, content: bytes) -> None:
        with tarfile.open(fileobj=(io.BytesIO(content)), mode='r:gz') as tar:
            for member in tar.getmembers():
                tar.extract(member, path=self._dir)


class VespaApplicationStore(ABC):
    """
    All Marqo index related operations need to change and redeploy VespaApplicationPackage. There are several ways to
    handle the change and deployment of the application. This class extracted an interface for low-level operations.
    """
    def __init__(self, vespa_client: VespaClient, deploy_timeout: int, wait_for_convergence_timeout: int):
        self._deploy_timeout = deploy_timeout
        self._wait_for_convergence_timeout = wait_for_convergence_timeout
        self._vespa_client = vespa_client

    @abstractmethod
    def file_exists(self, *paths: str) -> bool:
        """
        Check if a file exists at the given paths in the application package

        Args:
            *paths (str): The relative paths to check

        Returns:
            bool: True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def read_text_file(self, *paths: str) -> Optional[str]:
        """
        Return the content of a text file at the given paths in the application package

        Args:
            *paths (str): The relative paths to read

        Returns:
            str: The content of a text file at the given paths in the application package
            None if the file does not exist
        """
        pass

    @abstractmethod
    def read_binary_file(self, *paths: str) -> Optional[bytes]:
        """
        Return the content of a binary file at the given paths in the application package

        Args:
            *paths (str): The relative paths to read

        Returns:
            bytes: The content of a text file at the given paths in the application package
            None if the file does not exist
        """
        pass

    @abstractmethod
    def save_file(self, content: Union[str, bytes], *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        """
        Save the content to a file at the given paths in the application package

        Args:
            content (Union[str, bytes]): The content to save
            *paths (str): The relative paths to save
            backup (Optional[VespaAppBackup]): If provided, backup the old file
        """
        pass

    @abstractmethod
    def remove_file(self, *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        """
        Remove the content of a file at the given paths in the application package

        Args:
            *paths (str): The relative paths to remove
            backup (Optional[VespaAppBackup]): If provided, backup the old file
        """
        pass

    @abstractmethod
    def deploy_application(self) -> None:
        """
        Deploy the application package
        """
        pass


class VespaApplicationFileStore(VespaApplicationStore):
    """
    This implementation handles the deployment of a Vespa application package by downloading all contents in one
    deployment session and deploying the updated app in another deployment session by using prepareandactivate endpoint
    of Vespa deployment API. See https://docs.vespa.ai/en/reference/deploy-rest-api-v2.html#prepareandactivate for
    more details. This is the only viable option to deploy changes of binary files before Vespa version 8.382.22.
    We implement this approach to support bootstrapping and rollback for Vespa version prior to 8.382.22.
    """
    def __init__(self, vespa_client: VespaClient, deploy_timeout: int, wait_for_convergence_timeout: int):
        super().__init__(vespa_client, deploy_timeout, wait_for_convergence_timeout)
        self._app_root_path = vespa_client.download_application(check_for_application_convergence=True)

    def _full_path(self, *paths: str) -> str:
        return os.path.join(self._app_root_path, *paths)

    def file_exists(self, *paths: str) -> bool:
        return os.path.exists(self._full_path(*paths))

    def read_text_file(self, *paths: str) -> Optional[str]:
        if not self.file_exists(*paths):
            return None
        with open(self._full_path(*paths), 'r') as file:
            return file.read()

    def read_binary_file(self, *paths: str) -> Optional[bytes]:
        if not self.file_exists(*paths):
            return None
        with open(self._full_path(*paths), 'rb') as file:
            return file.read()

    def save_file(self, content: Union[str, bytes], *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        path = self._full_path(*paths)
        if os.path.exists(path):
            logger.warn(f"{path} already exists in application package, overwriting")
            if backup is not None:
                backup.backup_file(self.read_binary_file(*paths), *paths)
        else:  # add file
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if backup is not None:
                backup.mark_for_removal(*paths)

        mode = 'w' if isinstance(content, str) else 'wb'
        with open(path, mode) as f:
            f.write(content)

    def remove_file(self, *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        path = self._full_path(*paths)
        if not os.path.exists(path):
            logger.warn(f"{path} does not exist in application package, nothing to delete")
        else:
            if backup is not None:
                backup.backup_file(self.read_binary_file(*paths), *paths)
            os.remove(path)

    def deploy_application(self) -> None:
        self._vespa_client.deploy_application(self._app_root_path, timeout=self._deploy_timeout)
        self._vespa_client.wait_for_application_convergence(timeout=self._wait_for_convergence_timeout)


class ApplicationPackageDeploymentSessionStore(VespaApplicationStore):
    """
    This implementation handles the deployment of a Vespa application package in the same deployment session.
    This is a preferred solution since it leverages the optimistic locking mechanism to avoid race conditions.
    See https://docs.vespa.ai/en/reference/deploy-rest-api-v2.html#create-session for more details.
    However, this approach does not support binary files for Vespa version prior to 8.382.22.
    """
    def __init__(self, vespa_client: VespaClient, deploy_timeout: int, wait_for_convergence_timeout: int):
        super().__init__(vespa_client, deploy_timeout, wait_for_convergence_timeout)
        self._content_base_url, self._prepare_url = vespa_client.create_deployment_session()
        self._all_contents = vespa_client.list_contents(self._content_base_url)

    def file_exists(self, *paths: str) -> bool:
        content_url = self._vespa_client.get_content_url(self._content_base_url, *paths)
        return content_url in self._all_contents

    def read_text_file(self, *paths: str) -> Optional[str]:
        if not self.file_exists(*paths):
            return None
        return self._vespa_client.get_text_content(self._content_base_url, *paths)

    def read_binary_file(self, *paths: str) -> Optional[bytes]:
        if not self.file_exists(*paths):
            return None
        return self._vespa_client.get_binary_content(self._content_base_url, *paths)

    def save_file(self, content: Union[str, bytes], *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        if backup is not None:
            if self.file_exists(*paths):
                backup.backup_file(self.read_binary_file(*paths), *paths)
            else:
                backup.mark_for_removal(*paths)

        self._vespa_client.put_content(self._content_base_url, content, *paths)

    def remove_file(self, *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        if self.file_exists(*paths):
            if backup is not None:
                backup.backup_file(self.read_binary_file(*paths), *paths)
            self._vespa_client.delete_content(self._content_base_url, *paths)

    def deploy_application(self) -> None:
        prepare_response = self._vespa_client.prepare(self._prepare_url, timeout=self._deploy_timeout)
        # TODO handle prepare configChangeActions
        # https://docs.vespa.ai/en/reference/deploy-rest-api-v2.html#prepare-session
        self._vespa_client.activate(prepare_response['activate'], timeout=self._deploy_timeout)
        self._vespa_client.wait_for_application_convergence(timeout=self._wait_for_convergence_timeout)


class VespaApplicationPackage:
    """
    Represents a Vespa application package. This class provides useful methods to manage contents in the application
    package. A Vespa application package usually contains the following contents
    app-root
      | -- services.xml   # contains configuration for container components, and schemas in content nodes
      | -- hosts.xml
      | -- schemas
             | -- schema_name1.sd
             \ -- marqo__settings.sd    # this is used to store index settings prior to v2.13.0
      | -- search
             \ -- query-profiles
                    \ -- default.xml    # default query profile
      | -- components
             \ -- marqo-custom-searchers-deploy.jar
      | -- validation-overrides.xml
      | -- marqo_index_settings.json          # NEW, stores index settings after v2.13.0
      | -- marqo_index_settings_history.json  # NEW, stores history of index
      \ -- marqo_config.json                  # NEW, stores marqo config (version info)
    """

    _SERVICES_XML_FILE = 'services.xml'
    _MARQO_CONFIG_FILE = 'marqo_config.json'
    _MARQO_INDEX_SETTINGS_FILE = 'marqo_index_settings.json'
    _MARQO_INDEX_SETTINGS_HISTORY_FILE = 'marqo_index_settings_history.json'
    _BACKUP_FILE = 'app_bak.tgz'
    _COMPONENTS_JAR_FOLDER = Path(__file__).parent / '../../../../vespa/target'

    def __init__(self, store: VespaApplicationStore):
        self._store = store
        # Mark the app package as configured if Marqo config json file exist
        self.is_configured = self._store.file_exists(self._MARQO_CONFIG_FILE)
        self._service_xml = ServicesXml(self._store.read_text_file(self._SERVICES_XML_FILE))
        self._marqo_config_store = MarqoConfigStore(self._store.read_text_file(self._MARQO_CONFIG_FILE))
        self._index_setting_store = IndexSettingStore(
            self._store.read_text_file(self._MARQO_INDEX_SETTINGS_FILE) or '{}',
            self._store.read_text_file(self._MARQO_INDEX_SETTINGS_HISTORY_FILE) or '{}')

    def get_marqo_config(self) -> MarqoConfig:
        return self._marqo_config_store.get()

    def bootstrap(self, to_version: str, existing_index_settings: Optional[List[MarqoIndex]]) -> None:
        """
        Bootstrap the Vespa application package to match the Marqo version. This will
        1. Migrate the existing index settings (once-off for the first time)
        2. Configure query profiles
        3. Configure custom components and copy component jar files
        4. Update the MarqoConfig version to match Marqo version
        5. Create a backup zip file containing old version of changed files for rollback
        6. Deploy the application pacakge
        """
        logger.info(f'Bootstrapping the vector store to {to_version}')

        if not self._store.file_exists(self._MARQO_INDEX_SETTINGS_FILE):
            # Migrate existing index settings from previous version of Marqo. This migration is a once-off operation.
            # It will be skipped if the index setting json files exists after the first bootstrapping.
            if existing_index_settings:
                logger.debug(f'Migrating existing index settings {[index.json for index in existing_index_settings]}')
                for index in existing_index_settings:
                    self._index_setting_store.save_index_setting(index)

            logger.debug(f'Persisting index settings: {self._index_setting_store.to_json()}')
            # please note that the index settings are not saved in backup since rolling back index settings might
            # cause corruption of an index.
            self._persist_index_settings()

        # A backup is created and passed to all the places a config file is updated or added, so that the older
        # version of the config can be backed up, or marked as need-to-remove when rolling back.
        backup = VespaAppBackup()
        self._configure_query_profiles(backup)
        self._copy_components_jar()  # we do not back up jar file
        self._service_xml.config_components()
        self._marqo_config_store.update_version(to_version)

        logger.debug(f'Persisting services.xml file: {self._service_xml}')
        self._store.save_file(self._service_xml.to_xml(), self._SERVICES_XML_FILE, backup=backup)

        logger.debug(f'Persisting marqo config: {self._marqo_config_store.get().json()}')
        self._store.save_file(self._marqo_config_store.get().json(), self._MARQO_CONFIG_FILE, backup=backup)

        logger.debug(f'Generating backup file {self._BACKUP_FILE} from {backup}')
        self._store.save_file(backup.to_zip_stream().read(), self._BACKUP_FILE)
        self._deploy()

    def rollback(self, marqo_version: str) -> None:
        """
        Rollback the Vespa application package to match the current Marqo version. This will
        1. Check if rollback is feasible (version matches, backup file exists and no irreversible changes)
        2. Recover the files from backup and remove the added files (marked as removal in backup)
        3. Create a backup of this rollback
        """
        vespa_app_version = self.get_marqo_config().version
        marqo_sem_version = semver.VersionInfo.parse(marqo_version, optional_minor_and_patch=True)
        app_sem_version = semver.VersionInfo.parse(vespa_app_version, optional_minor_and_patch=True)
        if marqo_sem_version >= app_sem_version:
            raise ApplicationRollbackError(f"Cannot rollback from ${app_sem_version} to ${marqo_sem_version}. "
                                           f"The target version must be lower than the current one.")

        if not self._store.file_exists(self._BACKUP_FILE):
            raise ApplicationRollbackError(f"{self._BACKUP_FILE} does not exist in current session, failed to rollback")

        old_backup = VespaAppBackup(self._store.read_binary_file(self._BACKUP_FILE))
        logger.debug(f'Old backup {old_backup}')

        marqo_config = MarqoConfigStore(old_backup.read_text_file(self._MARQO_CONFIG_FILE)).get()
        if marqo_config is not None and marqo_config.version != marqo_version:
            raise ApplicationRollbackError(f"Cannot rollback to {marqo_config.version}, current Marqo version is {marqo_version}")

        logger.info(f'Rolling the vector store back from {vespa_app_version} to {marqo_version}')

        # Copy components jar files from the old Marqo version
        self._copy_components_jar()

        new_backup = VespaAppBackup()
        for paths in old_backup.files_to_remove():
            logger.debug(f'Removing file {paths}')
            self._store.remove_file(*paths, backup=new_backup)

        for (path, file_content) in old_backup.files_to_rollback():
            logger.debug(f'Rolling back file {path}')
            if path == self._SERVICES_XML_FILE:
                self._validate_services_xml_for_rollback(file_content.decode('utf-8'))
            self._store.save_file(file_content, path, backup=new_backup)

        logger.debug(f'Generating backup file {self._BACKUP_FILE} from {new_backup}')
        self._store.save_file(new_backup.to_zip_stream().read(), self._BACKUP_FILE)
        self._deploy()

    def batch_add_index_setting_and_schema(self, indexes: List[Tuple[str, MarqoIndex]]) -> None:
        for schema, index in indexes:
            if self.has_index(index.name):
                raise IndexExistsError(f"Index {index.name} already exists")

            self._index_setting_store.save_index_setting(index)
            self._store.save_file(schema, 'schemas', f'{index.schema_name}.sd')
            self._service_xml.add_schema(index.schema_name)

        self._persist_index_settings()
        self._store.save_file(self._service_xml.to_xml(), self._SERVICES_XML_FILE)
        self._deploy()

    def batch_delete_index_setting_and_schema(self, index_names: List[str]) -> None:
        for name in index_names:
            index = self._index_setting_store.get_index(name)
            if index is None:
                raise IndexNotFoundError(f"Index {name} not found")
            self._index_setting_store.delete_index_setting(index.name)
            self._store.remove_file('schemas', f'{index.schema_name}.sd')
            self._service_xml.remove_schema(index.schema_name)

        self._add_schema_removal_override()
        self._persist_index_settings()
        self._store.save_file(self._service_xml.to_xml(), self._SERVICES_XML_FILE)
        self._deploy()

    def update_index_setting_and_schema(self, index: MarqoIndex, schema: str) -> None:
        if not self.has_index(index.name):
            raise IndexNotFoundError(f"Index {index.name} not found")

        version = index.version + 1 if index.version is not None else 1
        self._store.save_file(schema, 'schemas', f'{index.schema_name}.sd')
        self._index_setting_store.save_index_setting(index.copy(update={'version': version}))
        self._persist_index_settings()
        self._deploy()

    def has_schema(self, name: str) -> bool:
        return self._store.file_exists('schemas', f'{name}.sd')

    def has_index(self, name: str) -> bool:
        return self._index_setting_store.get_index(name) is not None

    def _deploy(self):
        self._store.deploy_application()

    def _persist_index_settings(self):
        index_setting_json, index_setting_history_json = self._index_setting_store.to_json()
        self._store.save_file(index_setting_json, self._MARQO_INDEX_SETTINGS_FILE)
        self._store.save_file(index_setting_history_json, self._MARQO_INDEX_SETTINGS_HISTORY_FILE)

    def _configure_query_profiles(self, backup: VespaAppBackup) -> None:
        content = textwrap.dedent(
            '''
            <query-profile id="default">
                <field name="maxHits">1000000</field>
                <field name="maxOffset">1000000</field>
            </query-profile>
            '''
        )
        logger.debug(f'Configuring query profiles {content}')
        self._store.save_file(content, 'search', 'query-profiles', 'default.xml', backup=backup)

    def _copy_components_jar(self) -> None:
        components_jar_file = 'marqo-custom-searchers-deploy.jar'

        logger.debug(f'Copying components jar file {components_jar_file}')
        with open(self._COMPONENTS_JAR_FOLDER/components_jar_file, 'rb') as f:
            self._store.save_file(f.read(), 'components', components_jar_file)

    def _add_schema_removal_override(self) -> None:
        content = textwrap.dedent(
            f'''
            <validation-overrides>
                 <allow until='{datetime.utcnow().strftime('%Y-%m-%d')}'>schema-removal</allow>
            </validation-overrides>
            '''
        ).strip()
        self._store.save_file(content, 'validation-overrides.xml')

    def _validate_services_xml_for_rollback(self, services_xml_backup: str) -> None:
        services_xml_old = ServicesXml(services_xml_backup)
        elements_to_check = [
            # (xml_path, error_message)
            ('content/documents', 'Indexes have been added or removed since last backup.'),
            ('*/nodes', 'Vector store config has been changed since the last backup.'),
            ('admin', 'Vector store config has been changed since the last backup.'),
        ]

        for (xml_path, error_message) in elements_to_check:
            if not self._service_xml.compare_element(services_xml_old, xml_path):
                logger.debug(f'{error_message} services.xml in backup: {services_xml_old} vs. '
                             f'current service.xml: {self._service_xml}')
                raise ApplicationRollbackError(f'Aborting rollback. Reason: {error_message}')
