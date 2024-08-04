import json
import os
import io
import tarfile
import tempfile
import textwrap
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Tuple, Generator

import httpx
import semver
from datetime import datetime

from pathlib import Path
import xml.etree.ElementTree as ET

from marqo.base_model import ImmutableBaseModel
from marqo.core.exceptions import InternalError, OperationConflictError, IndexNotFoundError, IndexExistsError
from marqo.core.models import MarqoIndex
import marqo.logging
from marqo.vespa.exceptions import VespaError
from marqo.vespa.vespa_client import VespaClient


logger = marqo.logging.get_logger(__name__)


class ServiceXml:
    """
    Represents a Vespa service XML file.
    """
    def __init__(self, xml_str: str):
        self._root = ET.fromstring(xml_str)
        self._documents = self._ensure_only_one('content/documents')

    def __str__(self) -> str:
        return self.to_xml()

    def to_xml(self) -> str:
        return ET.tostring(self._root).decode('utf-8')

    def _ensure_only_one(self, xml_path: str) -> ET.Element:
        elements = self._root.findall(xml_path)

        if len(elements) > 1:
            raise InternalError(f'Multiple {xml_path} elements found in services.xml. Only one is allowed')
        if len(elements) == 0:
            raise InternalError(f'No {xml_path} element found in services.xml')

        return elements[0]

    def add_schema(self, name: str) -> None:
        if self._documents.find(f'document[@type="{name}"]'):
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

    @staticmethod
    def _add_component(parent: ET.Element, tag: str, name: str, bundle: str = 'marqo-custom-components'):
        element = ET.SubElement(parent, tag)
        element.set('id', name)
        element.set('bundle', bundle)
        return element


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
        self._index_settings = {key: MarqoIndex.parse_obj(value) for key, value in json.loads(index_settings_json).items()}
        self._index_settings_history = {key: [MarqoIndex.parse_obj(value) for value in history_array]
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
                                             f'Current version {current_version}, new version {target_version}.')
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
    _REMOVE_FILE_LIST = "files_to_remove.json"

    def __init__(self, backup_zip_file_content: Optional[bytes] = None) -> None:
        self._dir = tempfile.mkdtemp()
        self._removal_mark_file = os.path.join(self._dir, self._REMOVE_FILE_LIST)
        self._files_to_remove = []

        if backup_zip_file_content is not None:
            self._extract_gzip_from_bytes(backup_zip_file_content)
            if os.path.isfile(self._removal_mark_file):
                self._files_to_remove = json.load(open(self._removal_mark_file))
                os.remove(self._removal_mark_file)

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
                    yield os.path.split(relpath), f.read()

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

    @abstractmethod
    def file_exists(self, *paths: str) -> bool:
        pass

    @abstractmethod
    def read_text_file(self, *paths: str) -> Optional[str]:
        pass

    @abstractmethod
    def read_binary_file(self, *paths: str) -> Optional[bytes]:
        pass

    @abstractmethod
    def save_file(self, content: Union[str, bytes], *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        pass

    @abstractmethod
    def remove_file(self, *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        pass


class VespaApplicationFileStore(VespaApplicationStore):
    def __init__(self, app_root_path: str):
        self._app_root_path = app_root_path

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


class ApplicationPackageDeploymentSessionStore(VespaApplicationStore):
    def __init__(self, session_id: int, http_client: httpx.Client, vespa_client: VespaClient):
        self._session_id = session_id
        self._http_client = http_client
        self._vespa_client = vespa_client
        self._all_contents = vespa_client.list_contents(session_id, http_client)

    def file_exists(self, *paths: str) -> bool:
        content_url = self._vespa_client.get_content_url(self._session_id, *paths)
        return content_url in self._all_contents

    def read_text_file(self, *paths: str) -> Optional[str]:
        if not self.file_exists(*paths):
            return None
        return self._vespa_client.get_text_content(self._session_id, self._http_client, *paths)

    def read_binary_file(self, *paths: str) -> Optional[bytes]:
        if not self.file_exists(*paths):
            return None
        return self._vespa_client.get_binary_content(self._session_id, self._http_client, *paths)

    def save_file(self, content: Union[str, bytes], *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        """
        Saves the given content to the given path in the application package. Please note that the binary content
        is not supported due to this Vespa bug: https://github.com/vespa-engine/vespa/issues/32016
        """
        if isinstance(content, bytes):
            raise VespaError("Uploading binary content to Vespa deployment session is currently not supported")
        if backup is not None:
            if not self.file_exists(*paths):
                backup.backup_file(self.read_binary_file(*paths), *paths)
            else:
                backup.mark_for_removal(*paths)

        self._vespa_client.put_content(self._session_id, self._http_client, content, *paths)

    def remove_file(self, *paths: str, backup: Optional[VespaAppBackup] = None) -> None:
        if self.file_exists(*paths):
            if backup is not None:
                backup.backup_file(self.read_binary_file(*paths), *paths)
            self._vespa_client.delete_content(self._session_id, self._http_client, *paths)


class VespaApplicationPackage:
    """
    Represents a Vespa application package. This class provides useful methods to manage contents in the application
    package. A Vespa application package usually contains the following contents
    app-root
      | -- services.xml   # contains configuration for container components, and schemas in content nodes
      | -- hosts.xml
      | -- schemas
             | -- schema_name1.sd
             \ -- marqo__settings.sd    # this is used to store index settings prior to v2.12.0
      | -- search
             \ -- query-profiles
                    \ -- default.xml    # default query profile
      | -- components
             \ -- marqo-custom-searchers-deploy.jar
      | -- validation-overrides.xml
      | -- marqo_index_settings.json          # NEW, stores index settings after v2.12.0
      | -- marqo_index_settings_history.json  # NEW, stores history of index
      \ -- marqo_config.json                  # NEW, stores marqo config (version info)
    """

    _SERVICES_XML_FILE = 'services.xml'
    _MARQO_CONFIG_FILE = 'marqo_config.json'
    _MARQO_INDEX_SETTINGS_FILE = 'marqo_index_settings.json'
    _MARQO_INDEX_SETTINGS_HISTORY_FILE = 'marqo_index_settings_history.json'
    _BACKUP_FILE = 'app_bak.zip'

    def __init__(self, store: VespaApplicationStore):
        self._store = store
        self.is_configured = self._store.file_exists(self._MARQO_CONFIG_FILE)
        self._service_xml = ServiceXml(self._store.read_text_file(self._SERVICES_XML_FILE))
        self._marqo_config_store = MarqoConfigStore(self._store.read_text_file(self._MARQO_CONFIG_FILE))
        self._index_setting_store = IndexSettingStore(
            self._store.read_text_file(self._MARQO_INDEX_SETTINGS_FILE) or '{}',
            self._store.read_text_file(self._MARQO_INDEX_SETTINGS_HISTORY_FILE) or '{}')

    def get_marqo_config(self) -> MarqoConfig:
        return self._marqo_config_store.get()

    def need_bootstrapping(self, marqo_version: str, marqo_config_doc: Optional[MarqoConfig] = None) -> bool:
        """
        Bootstrapping is needed when
        - the version of Marqo is higher than the version of the deployed application
        - the version of Marqo is lower but allow downgrade is enabled (used for rollback)
        The version of the application is retrieved in the following order:
        - from the 'marqo_config.json' file (Post v2.12.0)
        - from the marqo_config_doc passed in which is marqo__config doc saved in marqo__settings schema (Post v2.1.0)
        - if neither is available, return 2.0.0 as default (Pre v2.1.0 or not bootstrapped yet)
        """
        if self.is_configured and self.get_marqo_config() is not None:
            app_version = self.get_marqo_config().version
        elif marqo_config_doc:
            app_version = marqo_config_doc.version
        else:
            app_version = '2.0.0'

        marqo_sem_version = semver.VersionInfo.parse(marqo_version, optional_minor_and_patch=True)
        app_sem_version = semver.VersionInfo.parse(app_version, optional_minor_and_patch=True)

        return app_sem_version < marqo_sem_version

    def bootstrap(self, marqo_version: str, existing_index_settings: List[MarqoIndex] = ()) -> None:
        # Migrate existing index settings from previous versions of Marqo
        if not self.is_configured:
            for index in existing_index_settings:
                self._index_setting_store.save_index_setting(index)
            self._persist_index_settings()

        backup = VespaAppBackup()
        self._config_query_profiles(backup)
        self._copy_components_jar()  # we do not back jar file
        self._service_xml.config_components()
        self._marqo_config_store.update_version(marqo_version)

        self._store.save_file(self._service_xml.to_xml(), self._SERVICES_XML_FILE, backup=backup)
        self._store.save_file(self._marqo_config_store.get().json(), self._MARQO_CONFIG_FILE, backup=backup)
        self._store.save_file(backup.to_zip_stream().read(), self._BACKUP_FILE)

    def _persist_index_settings(self):
        index_setting_json, index_setting_history_json = self._index_setting_store.to_json()
        self._store.save_file(index_setting_json, self._MARQO_INDEX_SETTINGS_FILE)
        self._store.save_file(index_setting_history_json, self._MARQO_INDEX_SETTINGS_HISTORY_FILE)

    def rollback(self, marqo_version: str) -> bool:
        if not self._store.file_exists(self._BACKUP_FILE):
            logger.error(f"{self._BACKUP_FILE} does not exist in current session, failed to rollback")
            return False

        old_backup = VespaAppBackup(self._store.read_binary_file(self._BACKUP_FILE))

        marqo_config_json = old_backup.read_text_file(self._MARQO_CONFIG_FILE)
        rollback_version = MarqoConfigStore(marqo_config_json).get().version
        if rollback_version != marqo_version:
            logger.warn(f"Cannot rollback to {rollback_version}, current Marqo version is {marqo_version}")
            return False

        new_backup = VespaAppBackup()
        for paths in old_backup.files_to_remove():
            self._store.remove_file(*paths, backup=new_backup)

        for (paths, file_content) in old_backup.files_to_rollback():
            self._store.save_file(file_content, *paths, backup=new_backup)

        self._store.save_file(new_backup.to_zip_stream().read(), self._BACKUP_FILE)

    def _config_query_profiles(self, backup: VespaAppBackup) -> None:
        content = textwrap.dedent(
            '''
            <query-profile id="default">
                <field name="maxHits">1000000</field>
                <field name="maxOffset">1000000</field>
            </query-profile>
            '''
        )
        self._store.save_file(content, 'search', 'query-profiles', 'default.xml', backup=backup)

    def _copy_components_jar(self) -> None:
        vespa_jar_folder = Path(__file__).parent / '../../../../vespa/target'
        components_jar_files = ['marqo-custom-components-deploy.jar']

        # copy the components jar files to the empty folder
        for file in components_jar_files:
            with open(vespa_jar_folder/file, 'rb') as f:
                self._store.save_file(f.read(), 'components', file)

    def batch_add_index_setting_and_schema(self, indexes: List[Tuple[str, MarqoIndex]]) -> None:
        for schema, index in indexes:
            if self.has_index(index.name):
                raise IndexExistsError(f"Index {index.name} already exists")

            self._index_setting_store.save_index_setting(index)
            self._store.save_file(schema, 'schemas', f'{index.schema_name}.sd')
            self._service_xml.add_schema(index.schema_name)

        self._persist_index_settings()
        self._store.save_file(self._service_xml.to_xml(), self._SERVICES_XML_FILE)

    def delete_index_setting_and_schema(self, index_name: str) -> None:
        self.batch_delete_index_setting_and_schema([index_name])

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

    def has_schema(self, name: str) -> bool:
        return self._store.file_exists('schemas', f'{name}.sd')

    def has_index(self, name: str) -> bool:
        return self._index_setting_store.get_index(name) is not None

    def _add_schema_removal_override(self) -> None:
        content = textwrap.dedent(
            f'''
            <validation-overrides>
                 <allow until='{datetime.utcnow().strftime('%Y-%m-%d')}'>schema-removal</allow>
            </validation-overrides>
            '''
        ).strip()
        self._store.save_file(content, 'validation-overrides.xml')
