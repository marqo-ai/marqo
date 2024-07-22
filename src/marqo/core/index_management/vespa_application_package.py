import json
import os
import textwrap
import semver
from datetime import datetime

from marqo.core.exceptions import InternalError
from marqo.core.models import MarqoIndex
from marqo.vespa.vespa_client import VespaClient
import marqo.logging
import marqo.vespa.vespa_client
import xml.etree.ElementTree as ET

logger = marqo.logging.get_logger(__name__)


class ServiceXml:
    """
    Represents a Vespa service XML file.
    """
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise InternalError('Could not find file %s' % path)
        self._path = path
        self._tree = ET.parse(path)
        self._root = self._tree.getroot()
        self._documents = self._ensure_only_one('content/documents')

    def _save(self) -> None:
        self._tree.write(self._path)

    def _ensure_only_one(self, xml_path: str):
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
            self._save()

    def remove_schema(self, name: str) -> None:
        docs = self._documents.findall(f'document[@type="{name}"]')
        if not docs:
            logger.warn(f"Schema {name} does not exist in services.xml, nothing to remove")
        else:
            for doc in docs:
                self._documents.remove(doc)
            self._save()


class IndexSettingStore:
    """
    Index settings are now stored in a json file within the application package. Up to 3 historical versions of each
    index setting are stored in another json file. This class handles the creation and management of index settings
    """
    _HISTORY_VERSION_LIMIT = 3

    def __init__(self, index_settings_json: str, index_settings_history_json: str):
        self._index_settings = {key: MarqoIndex.parse_obj(value) for key, value in json.loads(index_settings_json).items()}
        self._index_settings_history = {key: [MarqoIndex.parse_obj(value) for value in history_array]
                                        for key, history_array in json.loads(index_settings_history_json).items()}

    def save_to_files(self, json_file: str, history_json_file: str) -> None:
        with open(json_file, 'w') as f:
            # Pydantic 1.x does not support creating jsonable dict. `json.loads(index.json())` is a workaround
            json.dump({key: json.loads(index.json()) for key, index in self._index_settings.items()}, f)
        with open(history_json_file, 'w') as f:
            json.dump({key: [json.loads(index.json()) for index in history] for
                       key, history in self._index_settings_history.items()}, f)

    def save_index_setting(self, index_setting: MarqoIndex) -> None:
        if index_setting.name in self._index_settings:
            # update the index setting, TODO check if the version matches first
            self._move_to_history(index_setting.name)
        else:
            # create a new index setting with version 1, TODO check if the version is actually 1
            if index_setting.name in self._index_settings_history:
                del self._index_settings_history[index_setting.name]
            self._index_settings[index_setting.name] = index_setting

    def delete_index_setting(self, index_setting_name: str) -> None:
        if index_setting_name not in self._index_settings:
            logger.warn(f"Index setting {index_setting_name} does not exist, nothing to delete")
        else:
            self._move_to_history(index_setting_name)
            del self._index_settings[index_setting_name]

    def _move_to_history(self, index_setting_name):
        if index_setting_name in self._index_settings_history:
            self._index_settings_history[index_setting_name].insert(0, self._index_settings[index_setting_name])
            self._index_settings_history[index_setting_name] = self._index_settings_history[index_setting_name][
                                                               :self._HISTORY_VERSION_LIMIT]
        else:
            self._index_settings_history[index_setting_name] = [self._index_settings[index_setting_name]]


class VespaApplicationPackage:
    """
    Represents a Vespa application package. Downloads the application package from Vespa when initialised.
    Provides convenient methods to manage various configs to the application package.
    """

    def __init__(self, vespa_client: VespaClient):
        self._vespa_client = vespa_client
        self._app_root_path = vespa_client.download_application(check_for_application_convergence=True)
        self._service_xml = ServiceXml(os.path.join(self._app_root_path, 'services.xml'))
        self._index_setting_store = IndexSettingStore(
            self._load_json_file(os.path.join(self._app_root_path, 'marqo_index_settings.json')),
            self._load_json_file(os.path.join(self._app_root_path, 'marqo_index_settings_history.json')))

    @staticmethod
    def _load_json_file(path: str, default_value: str = '{}') -> str:
        if not os.path.exists(path):
            return default_value
        with open(path, 'r') as file:
            return file.read()

    def deploy(self, deployment_timeout=60, convergence_timeout=120) -> None:
        self._index_setting_store.save_to_files(os.path.join(self._app_root_path, 'marqo_index_settings.json'),
                                                os.path.join(self._app_root_path, 'marqo_index_settings_history.json'))
        self._vespa_client.deploy_application(self._app_root_path, timeout=deployment_timeout)
        self._vespa_client.wait_for_application_convergence(timeout=convergence_timeout)

    def bootstrap(self, marqo_version: str, allow_downgrade: bool = False) -> bool:
        marqo_sem_version = semver.VersionInfo.parse(marqo_version, optional_minor_and_patch=True)
        app_sem_version = semver.VersionInfo.parse(self._get_version(), optional_minor_and_patch=True)

        if app_sem_version >= marqo_sem_version and not allow_downgrade:
            return False

        # TODO clean up the application package
        # remove components folder
        # remove query profile folder
        # remove customer component config from service.xml file

        # TODO bootstrap
        # copy components
        # generate query profiles
        # add customer component to service.xml
        # copy index settings if necessary
        # set version

        return True

    def _get_version(self) -> str:
        """
        Returns the version of the application package. This should match the version of Marqo which bootstraps the
        application package. Going forward, we will upgrade the application package in each Marqo release.
        It tries to retrieve version from the marqo_config.json file first. This file exists after v2.11.0
        If the file does not exist, it tries to get it from marqo__config doc in marqo__settings schema. This approach
        is used after v2.1.0. If it fails again, then return the default version 2.0.0
        """
        return '2.0.0'

    def add_schema(self, name: str, schema: str) -> None:
        self.save_txt_file(['schemas', f'{name}.sd'], schema)
        self._service_xml.add_schema(name)

    def remove_schema(self, name: str) -> None:
        self.delete_file(['schemas', f'{name}.sd'])
        self._service_xml.remove_schema(name)
        self._add_schema_removal_override()

    def _add_schema_removal_override(self) -> None:
        content = textwrap.dedent(
            f'''
            <validation-overrides>
                 <allow until='{datetime.utcnow().strftime('%Y-%m-%d')}'>schema-removal</allow>
            </validation-overrides>
            '''
        ).strip()
        self.save_txt_file(['validation-overrides.xml'], content)

    def update_components(self) -> None:
        pass

    def save_txt_file(self, file_path: [str], content: str) -> None:
        path = os.path.join(self._app_root_path, *file_path)
        if os.path.exists(path):
            logger.warn(f"{path} already exists in application package, overwriting")

        with open(path, 'w') as f:
            f.write(content)

    def delete_file(self, file_path: [str]) -> None:
        path = os.path.join(self._app_root_path, *file_path)
        if not os.path.exists(path):
            logger.warn(f"{path} does not exist in application package, nothing to delete")
        else:
            os.remove(path)
