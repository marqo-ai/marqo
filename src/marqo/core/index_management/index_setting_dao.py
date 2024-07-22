from typing import Protocol, List, Optional

from marqo.core import constants
from marqo.core.exceptions import InternalError
from marqo.core.index_management.vespa_application_package import VespaApplicationPackage
from marqo.core.models import MarqoIndex
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient


class IndexSettingDao(Protocol):
    def fetch_all(self) -> List[MarqoIndex]:
        ...

    def fetch_by_name(self, index_name: str) -> Optional[MarqoIndex]:
        ...

    def save(self, index_setting: MarqoIndex, app: VespaApplicationPackage) -> None:
        ...

    def delete(self, index_setting: MarqoIndex, app: VespaApplicationPackage) -> None:
        ...


class VespaDocumentIndexSettingDao:
    _MARQO_SETTINGS_SCHEMA_NAME = 'marqo__settings'

    def __init__(self, vespa_client: VespaClient):
        self.vespa_client = vespa_client

    def save(self, index_setting: MarqoIndex, app: VespaApplicationPackage) -> None:
        """
        Create or update index settings in Vespa settings schema.
        """
        self.vespa_client.feed_document(
            VespaDocument(
                id=index_setting.name,
                fields={
                    'index_name': index_setting.name,
                    'settings': index_setting.json()
                }
            ),
            schema=self._MARQO_SETTINGS_SCHEMA_NAME
        )

    def delete(self, index_setting: MarqoIndex, app: VespaApplicationPackage) -> None:
        app.remove_schema(index_setting.name)
        app.deploy()
        self.vespa_client.delete_document(index_setting.name, self._MARQO_SETTINGS_SCHEMA_NAME)

    def fetch_all(self) -> List[MarqoIndex]:
        """
        Get all Marqo indexes.

        Returns:
            List of Marqo indexes
        """
        batch_response = self.vespa_client.get_all_documents(self._MARQO_SETTINGS_SCHEMA_NAME, stream=True)
        if batch_response.continuation:
            # TODO - Verify expected behaviour when streaming. Do we need to expect and handle pagination?
            raise InternalError("Unexpected continuation token received")

        return [
            MarqoIndex.parse_raw(document.fields['settings'])

            for document in batch_response.documents
            if not document.id.split('::')[-1].startswith(constants.MARQO_RESERVED_PREFIX)
        ]

    def fetch_by_name(self, index_name: str) -> Optional[MarqoIndex]:
        """
        Get a Marqo index by name.

        Args:
            index_name: Name of Marqo index to get

        Returns:
            Marqo index
        """
        try:
            response = self.vespa_client.get_document(index_name, self._MARQO_SETTINGS_SCHEMA_NAME)
        except VespaStatusError as e:
            if e.status_code == 404:
                return None
            raise e

        return MarqoIndex.parse_raw(response.document.fields['settings'])


class VespaAppPkgIndexSettingDao:
    _MARQO_SETTINGS_FOLDER = 'marqo_index_settings'

    def __init__(self, vespa_client: VespaClient):
        self.vespa_client = vespa_client

    def fetch_all(self) -> List[MarqoIndex]:
        return self.vespa_client.get_all_index_settings()

    def fetch_by_name(self, index_name: str) -> Optional[MarqoIndex]:
        return self.vespa_client.get_index_setting_by_name(index_name)

    def save(self, index_setting: MarqoIndex, app: VespaApplicationPackage) -> None:
        file_path = [self._MARQO_SETTINGS_FOLDER, f'{index_setting.name}.json']
        app._save_txt_file(file_path, index_setting.json())

    def delete(self, index_setting: MarqoIndex, app: VespaApplicationPackage) -> None:
        file_path = [self._MARQO_SETTINGS_FOLDER, f'{index_setting.name}.json']
        app._delete_file(file_path)
