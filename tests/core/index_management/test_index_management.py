import json
import os
import textwrap
from unittest import mock
from datetime import datetime

import httpx

import xml.etree.ElementTree as ET


from marqo import version
from marqo.core.exceptions import IndexExistsError, ApplicationNotInitializedError, InternalError
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.index_management.vespa_application_package import MarqoConfig, VespaApplicationPackage, \
    ApplicationPackageDeploymentSessionStore
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.vespa.exceptions import VespaActivationConflictError
from marqo.vespa.models import VespaDocument
from tests.marqo_test import MarqoTestCase


class IndexResilienceError(Exception):
    """A custom exception to raise when an error is encountered during the index resilience test."""
    pass


class TestIndexManagement(MarqoTestCase):

    def setUp(self):
        super().setUp()
        self.index_management = IndexManagement(self.vespa_client,
                                                zookeeper_client=self.zookeeper_client,
                                                enable_index_operations=True)
        # this resets the application package to a clean state
        self._test_dir = os.path.dirname(os.path.abspath(__file__))
        self._deploy_initial_app_package()

    def test_clean_bootstrap_vespa(self):
        bootstrapped = self.index_management.bootstrap_vespa()
        self.assertTrue(bootstrapped)

        app = self.vespa_client.download_application()
        # TODO find a better way to test this, it assume the jar file is generated in target folder
        self._assert_file_exists(app, 'components', 'marqo-custom-components-deploy.jar')
        self._assert_file_exists(app, 'search', 'query-profiles', 'default.xml')
        self._assert_file_exists(app, 'marqo_index_settings.json')
        self._assert_file_exists(app, 'marqo_index_settings_history.json')
        self._assert_file_exists(app, 'marqo_config.json')

        # Verify no index setting is present
        with open(os.path.join(app, 'marqo_index_settings.json')) as f:
            self.assertEqual('{}', f.read())

        # Verify no index setting history is present
        with open(os.path.join(app, 'marqo_index_settings_history.json')) as f:
            self.assertEqual('{}', f.read())

        with open(os.path.join(app, 'marqo_config.json')) as f:
            self.assertEqual(json.loads(f'{{"version": "{version.get_version()}"}}'), json.load(f))

        self.assertEqual(self.index_management.get_marqo_version(), version.get_version())

    def test_skip_boostrap_if_already_bootstrapped(self):
        def modified_post(*args, **kwargs):
            return httpx.post(*args, **kwargs)

        # verify the first boostrap call deploys the app to vespa
        with mock.patch.object(httpx.Client, 'post', side_effect=modified_post) as mock_post:
            self.assertTrue(self.index_management.bootstrap_vespa())
            self.assertEqual(mock_post.call_count, 2)
            self.assertTrue('prepareandactivate' in mock_post.call_args_list[1].args[0])

        # verify the second boostrap call skips the deployment
        with mock.patch.object(httpx.Client, 'post', side_effect=modified_post) as mock_post:
            self.assertFalse(self.index_management.bootstrap_vespa())
            self.assertEqual(mock_post.call_count, 1)
            self.assertFalse('prepareandactivate' in mock_post.call_args_list[0].args[0])

    def test_boostrap_from_existing_app_prior_to_2_12(self):
        existing_index = self._deploy_existing_app_package()
        existing_index_with_version = existing_index.copy(update={'version': 1})

        bootstrapped = self.index_management.bootstrap_vespa()
        self.assertTrue(bootstrapped)

        app = self.vespa_client.download_application()
        self._assert_file_exists(app, 'marqo_index_settings.json')
        self._assert_file_exists(app, 'marqo_index_settings_history.json')
        self._assert_file_exists(app, 'marqo_config.json')

        with open(os.path.join(app, 'marqo_index_settings.json')) as f:
            index_settings = json.load(f)
            self.assertTrue(existing_index.name in index_settings)
            self.assertEqual(existing_index_with_version.json(), json.dumps(index_settings[existing_index.name]))

        with open(os.path.join(app, 'marqo_index_settings_history.json')) as f:
            self.assertEqual('{}', f.read())

        with open(os.path.join(app, 'marqo_config.json')) as f:
            self.assertEqual(json.loads(f'{{"version": "{version.get_version()}"}}'), json.load(f))

    def test_bootstrap_overrides_component_jars_and_configs(self):
        pass

    def test_bootstrap_support_rollback(self):
        # TODO this feature is not implemented yet
        pass

    def test_distributed_lock(self):
        pass

    def test_prefixes(self):
        pass

    def test_index_operation_fails_if_disabled(self):
        # Create an index management instance with index operation disabled (by default)
        self.index_management = IndexManagement(self.vespa_client, zookeeper_client=None)
        index_request_1 = self.structured_marqo_index_request(
            fields=[FieldRequest(name='title', type=FieldType.Text)],
            tensor_fields=['title']
        )
        index_request_2 = self.unstructured_marqo_index_request()

        with self.assertRaises(InternalError):
            self.index_management.create_index(index_request_1)

        with self.assertRaises(InternalError):
            self.index_management.batch_create_indexes([index_request_1, index_request_2])

        with self.assertRaises(InternalError):
            self.index_management.delete_index_by_name(index_request_1.name)

        with self.assertRaises(InternalError):
            self.index_management.batch_delete_indexes_by_name([index_request_1.name, index_request_2.name])

    def test_index_operation_fails_if_not_bootstrapped(self):
        index_request_1 = self.structured_marqo_index_request(
            fields=[FieldRequest(name='title', type=FieldType.Text)],
            tensor_fields=['title']
        )
        index_request_2 = self.unstructured_marqo_index_request()

        with self.assertRaises(ApplicationNotInitializedError):
            self.index_management.create_index(index_request_1)

        with self.assertRaises(ApplicationNotInitializedError):
            self.index_management.batch_create_indexes([index_request_1, index_request_2])

        with self.assertRaises(ApplicationNotInitializedError):
            self.index_management.delete_index_by_name(index_request_1.name)

        with self.assertRaises(ApplicationNotInitializedError):
            self.index_management.batch_delete_indexes_by_name([index_request_1.name, index_request_2.name])

    def test_create_and_delete_index_successful(self):
        # merge batch create and delete happy path to save some testing time
        request = self.unstructured_marqo_index_request()
        schema, index = vespa_schema_factory(request).generate_schema()
        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        app = self.vespa_client.download_application()
        self._assert_index_is_present(app, index, schema)

        self.index_management.delete_index_by_name(index.name)

        app = self.vespa_client.download_application()
        self._assert_index_is_not_present(app, index.name, index.schema_name)

    def test_create_index_fails_if_exists(self):
        request = self.unstructured_marqo_index_request(name="test-index")
        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        with self.assertRaises(IndexExistsError):
            self.index_management.create_index(request)

    def test_delete_index_fails_when_index_not_found(self):
        self.index_management.bootstrap_vespa()

        with self.assertRaises(IndexNotFoundError):
            self.index_management.delete_index_by_name('index-does-not-exist')

    def test_batch_create_and_delete_index_successful(self):
        # merge batch create and delete happy path to save some testing time
        request1 = self.unstructured_marqo_index_request()
        request2 = self.structured_marqo_index_request(
            fields=[FieldRequest(name='title', type=FieldType.Text)],
            tensor_fields=['title']
        )
        schema1, index1 = vespa_schema_factory(request1).generate_schema()
        schema2, index2 = vespa_schema_factory(request2).generate_schema()

        self.index_management.bootstrap_vespa()
        self.index_management.batch_create_indexes([request1, request2])

        app = self.vespa_client.download_application()
        self._assert_index_is_present(app, index1, schema1)
        self._assert_index_is_present(app, index2, schema2)

        self.index_management.batch_delete_indexes_by_name([request1.name, request2.name])

        app = self.vespa_client.download_application()
        self._assert_index_is_not_present(app, index1.name, index1.schema_name)
        self._assert_index_is_not_present(app, index2.name, index2.schema_name)

    def test_batch_create_index_fails_atomically(self):
        request = self.unstructured_marqo_index_request(name="index1")
        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        request2 = self.unstructured_marqo_index_request(name="index2")
        _, index2 = vespa_schema_factory(request2).generate_schema()

        with self.assertRaises(IndexExistsError):
            self.index_management.batch_create_indexes([request2, request])

        app = self.vespa_client.download_application()
        self._assert_index_is_not_present(app, index2.name, index2.schema_name)

    def test_batch_delete_index_fails_atomically(self):
        request = self.unstructured_marqo_index_request(name="index1")
        schema, index1 = vespa_schema_factory(request).generate_schema()

        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        request2 = self.unstructured_marqo_index_request(name="index2")
        _, index2 = vespa_schema_factory(request2).generate_schema()

        with self.assertRaises(IndexNotFoundError):
            self.index_management.batch_delete_indexes_by_name([request.name, request2.name])

        app = self.vespa_client.download_application()
        self._assert_index_is_present(app, index1, schema)
        self._assert_index_is_not_present(app, index2.name, index2.schema_name)

    def test_race_condition(self):
        """
        In this test, we simulate two instances/threads of Marqo make different changes to the application package
        """
        self.index_management.bootstrap_vespa()

        request1 = self.unstructured_marqo_index_request()
        request2 = self.structured_marqo_index_request(
            fields=[FieldRequest(name='title', type=FieldType.Text)],
            tensor_fields=['title']
        )
        schema1, index1 = vespa_schema_factory(request1).generate_schema()
        schema2, index2 = vespa_schema_factory(request2).generate_schema()

        with self.vespa_client.deployment_session() as (session1, client1):
            store1 = ApplicationPackageDeploymentSessionStore(session1, client1, self.vespa_client)
            app1 = VespaApplicationPackage(store1)

            with self.vespa_client.deployment_session() as (session2, client2):
                store2 = ApplicationPackageDeploymentSessionStore(session2, client2, self.vespa_client)
                app2 = VespaApplicationPackage(store2)

                app1.add_index_setting_and_schema(index1, schema1)
                app1.save_to_store()

                app2.add_index_setting_and_schema(index2, schema2)
                app2.save_to_store()

                self.vespa_client.prepare(session1, client1)
                self.vespa_client.activate(session1, client1)

                self.vespa_client.prepare(session2, client2)
                # this should fail due to optimistic locking
                with self.assertRaises(VespaActivationConflictError):
                    self.vespa_client.activate(session2, client2)

    def test_os_path(self):
        print(os.path.join('.'))

    def _assert_index_is_present(self, app, expected_index, expected_schema):
        if 'version' not in expected_index:
            expected_index = expected_index.copy(update={'version': 1})

        # assert index setting exists and equals to expected value
        saved_index = self.index_management.get_index(expected_index.name)
        self.assertEqual(saved_index, expected_index)

        # assert schema file exists and has expected value
        schema_name = expected_index.schema_name
        self._assert_file_exists(app, 'schemas', f'{schema_name}.sd')
        with open(os.path.join(app, 'schemas', f'{schema_name}.sd')) as f:
            self.assertEqual(f.read(), expected_schema)
        doc = ET.parse(os.path.join(app, 'services.xml')).getroot().find(f'content/documents/document[@type="{schema_name}"]')
        self.assertIsNotNone(doc)

    def _assert_index_is_not_present(self, app, index_name, schema_name):
        with self.assertRaises(IndexNotFoundError):
            self.index_management.get_index(index_name)

        self._assert_file_does_not_exist(app, 'schemas', f'{schema_name}.sd')
        doc = ET.parse(os.path.join(app, 'services.xml')).getroot().find(
            f'content/documents/document[@type="{schema_name}"]')
        self.assertIsNone(doc)

    def _deploy_initial_app_package(self):
        app_root_path = os.path.join(self._test_dir, 'initial_vespa_app')
        self._add_schema_removal_override(app_root_path)
        self.vespa_client.deploy_application(app_root_path)
        self.vespa_client.wait_for_application_convergence()

    def _deploy_existing_app_package(self) -> MarqoIndex:
        app_root_path = os.path.join(self._test_dir, 'existing_vespa_app')
        self._add_schema_removal_override(app_root_path)
        marqo_index_request = self.unstructured_marqo_index_request(name="existing_index")
        schema, index = vespa_schema_factory(marqo_index_request).generate_schema()
        with open(os.path.join(app_root_path, 'schemas', f'{index.schema_name}.sd'), 'w') as f:
            f.write(schema)
        self.vespa_client.deploy_application(app_root_path)
        self.vespa_client.wait_for_application_convergence()
        self._save_index_settings(index)
        self._save_marqo_version('2.10.0')

        return index

    @staticmethod
    def _add_schema_removal_override(app_root_path: str):
        content = textwrap.dedent(
            f'''
            <validation-overrides>
                 <allow until='{datetime.utcnow().strftime('%Y-%m-%d')}'>schema-removal</allow>
            </validation-overrides>
            '''
        ).strip()
        with open(os.path.join(app_root_path, 'validation-overrides.xml'), 'w') as f:
            f.write(content)

    def _assert_file_exists(self, *file_paths):
        self.assertTrue(os.path.exists(os.path.join(*file_paths)), f'File {"/".join(file_paths[1:])} does not exist')

    def _assert_file_does_not_exist(self, *file_paths):
        self.assertFalse(os.path.exists(os.path.join(*file_paths)),f'File {"/".join(file_paths[1:])} exists')

    def _save_marqo_version(self, version: str) -> None:
        self.vespa_client.feed_document(
            VespaDocument(
                id=self.index_management._MARQO_CONFIG_DOC_ID,
                fields={
                    'settings': MarqoConfig(version=version).json()
                }
            ),
            schema=self.index_management._MARQO_SETTINGS_SCHEMA_NAME
        )

    def _save_index_settings(self, marqo_index: MarqoIndex) -> None:
        self.vespa_client.feed_document(
            VespaDocument(
                id=marqo_index.name,
                fields={
                    'index_name': marqo_index.name,
                    'settings': marqo_index.json()
                }
            ),
            schema=self.index_management._MARQO_SETTINGS_SCHEMA_NAME
        )
