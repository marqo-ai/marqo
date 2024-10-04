import filecmp
import json
import os
import tarfile
import tempfile
import textwrap
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import cast
from unittest import mock
from unittest.mock import patch

import httpx
import pytest

from marqo import version
from marqo.core.exceptions import IndexExistsError, ApplicationNotInitializedError, InternalError, \
    ApplicationRollbackError, OperationConflictError
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.index_management.vespa_application_package import (MarqoConfig, VespaApplicationPackage,
                                                                   ApplicationPackageDeploymentSessionStore)
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.vespa_index.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.s2_inference.s2_inference import get_model_properties_from_registry
from marqo.vespa.exceptions import VespaActivationConflictError
from marqo.vespa.models import VespaDocument
from tests.marqo_test import MarqoTestCase


@pytest.mark.slowtest
class TestIndexManagement(MarqoTestCase):

    def setUp(self):
        super().setUp()
        self.index_management = IndexManagement(self.vespa_client,
                                                zookeeper_client=self.zookeeper_client,
                                                enable_index_operations=True,
                                                deployment_timeout_seconds=30,
                                                convergence_timeout_seconds=60)
        # this resets the application package to a clean state
        self._test_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self._deploy_initial_app_package()

    def test_bootstrap_vespa_should_successfully_bootstrap_a_new_vespa_application_package(self):
        bootstrapped = self.index_management.bootstrap_vespa()
        self.assertTrue(bootstrapped)

        app = self.vespa_client.download_application()
        # TODO find a better way to test this, it assume the jar file is generated in target folder
        self._assert_file_exists(app, 'components', 'marqo-custom-searchers-deploy.jar')
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

    @patch('marqo.vespa.vespa_client.VespaClient.get_vespa_version')
    def test_bootstrap_vespa_should_successfully_bootstrap_a_new_vespa_application_package_with_old_vespa_version(
            self, mock_vespa_version):
        mock_vespa_version.return_value = '8.382.21'  # a fake version prior to minimum version supporting binary upload
        bootstrapped = self.index_management.bootstrap_vespa()
        self.assertTrue(bootstrapped)

        app = self.vespa_client.download_application()
        # TODO find a better way to test this, it assume the jar file is generated in target folder
        self._assert_file_exists(app, 'components', 'marqo-custom-searchers-deploy.jar')
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

    @patch('marqo.vespa.vespa_client.VespaClient.get_vespa_version')
    def test_bootstrap_vespa_should_skip_bootstrapping_if_already_bootstrapped_for_older_vespa_version(
            self, mock_vespa_version):
        mock_vespa_version.return_value = '8.382.21'

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

    def test_bootstrap_vespa_should_skip_bootstrapping_if_already_bootstrapped(self):
        def modified_put(*args, **kwargs):
            return httpx.put(*args, **kwargs)

        # verify the first boostrap call deploys the app to vespa
        with mock.patch.object(httpx.Client, 'put', side_effect=modified_put) as mock_post:
            self.assertTrue(self.index_management.bootstrap_vespa())
            self.assertTrue('prepare' in mock_post.call_args_list[-2].args[0])
            self.assertTrue('active' in mock_post.call_args_list[-1].args[0])

        # verify the second boostrap call skips the deployment
        with mock.patch.object(httpx.Client, 'put', side_effect=modified_put) as mock_post:
            self.assertFalse(self.index_management.bootstrap_vespa())
            self.assertEqual(mock_post.call_count, 0)

    def test_boostrap_vespa_should_migrate_index_settings_from_existing_vespa_app(self):
        """
        When we upgrade Marqo from prior to 2.13.0 to the latest version, we will migrate the index settings in the
        marqo__settings vespa schema to json files stored in the application package. This test cases tests if this
        migration is done correctly
        """
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

        # Verify no index setting history is present
        with open(os.path.join(app, 'marqo_index_settings_history.json')) as f:
            self.assertEqual('{}', f.read())

        with open(os.path.join(app, 'marqo_config.json')) as f:
            self.assertEqual(json.loads(f'{{"version": "{version.get_version()}"}}'), json.load(f))

    def test_bootstrap_vespa_should_override_and_backup_configs(self):
        self._deploy_existing_app_package()
        self.index_management.bootstrap_vespa()

        app = str(self.vespa_client.download_application())
        self._assert_file_exists(app, 'app_bak.tgz')
        backup_dir = tempfile.mkdtemp()
        with tarfile.open(os.path.join(app, 'app_bak.tgz'), mode='r:gz') as tar:
            for member in tar.getmembers():
                tar.extract(member, path=backup_dir)

        # Assert that following files are changed
        expected_updated_files = [
            ['services.xml'],
            ['components', 'marqo-custom-searchers-deploy.jar'],
            ['search', 'query-profiles', 'default.xml'],
        ]
        for file in expected_updated_files:
            self._assert_files_not_equal(
                os.path.join(app, *file),
                os.path.join(self._test_dir, 'existing_vespa_app', *file)
            )

        # Assert that following files are backed up, note that binary files won't be backed up
        expected_backup_files = [
            ['services.xml'],
            ['search', 'query-profiles', 'default.xml'],
        ]
        for file in expected_backup_files:
            self._assert_files_equal(
                os.path.join(backup_dir, *file),
                os.path.join(self._test_dir, 'existing_vespa_app', *file)
            )

    def test_rollback_should_succeed(self):
        self._deploy_existing_app_package()
        self.index_management.bootstrap_vespa()

        latest_version = str(self.vespa_client.download_application())

        # before we roll back, we'll mock the app session to use the previous version and jar files
        components_jar_folder = Path(__file__).parent / 'existing_vespa_app' / 'components'
        with mock.patch.object(VespaApplicationPackage, '_COMPONENTS_JAR_FOLDER', components_jar_folder):
            with mock.patch.object(version, 'get_version', return_value='2.10.0'):
                self.index_management.rollback_vespa()

        rolled_back_version = str(self.vespa_client.download_application())
        # Test the rollback rolls back the configs and component jar files to previous version
        expected_rolled_back_files = [
            ['services.xml'],
            ['components', 'marqo-custom-searchers-deploy.jar'],
            ['search', 'query-profiles', 'default.xml'],
        ]
        for file in expected_rolled_back_files:
            self._assert_files_equal(
                os.path.join(rolled_back_version, *file),
                os.path.join(self._test_dir, 'existing_vespa_app', *file)
            )
        # marqo_config.json does not exist in the previous version, and it gets deleted
        self._assert_file_does_not_exist(rolled_back_version, 'marqo_config.json')

        # rollback backs up the content in the latest version,
        self._assert_file_exists(rolled_back_version, 'app_bak.tgz')
        backup_dir = tempfile.mkdtemp()
        with tarfile.open(os.path.join(rolled_back_version, 'app_bak.tgz'), mode='r:gz') as tar:
            for member in tar.getmembers():
                tar.extract(member, path=backup_dir)

        # Test the rollback backs up file in the latest version
        expected_backup_files = [
            ['services.xml'],
            ['search', 'query-profiles', 'default.xml'],
        ]
        for file in expected_backup_files:
            self._assert_files_equal(
                os.path.join(backup_dir, *file),
                os.path.join(latest_version, *file)
            )

    def test_rollback_should_fail_when_target_version_is_current_version(self):
        self.index_management.bootstrap_vespa()
        with self.assertRaisesStrict(ApplicationRollbackError) as e:
            self.index_management.rollback_vespa()
        self.assertIn("The target version must be lower than the current one", str(e.exception))

    def test_rollback_should_fail_when_target_version_does_not_match_backup_version(self):
        with mock.patch.object(version, 'get_version', return_value='2.12.0'):
            self.index_management.bootstrap_vespa()  # writes 2.12.0 to marqo_config
        with mock.patch.object(version, 'get_version', return_value='2.14.0'):
            self.index_management.bootstrap_vespa()  # backs up 2.12.0

        with mock.patch.object(version, 'get_version', return_value='2.13.0'):
            # rolling back to 2.13.0 should raise error
            with self.assertRaisesStrict(ApplicationRollbackError) as e:
                self.index_management.rollback_vespa()
            self.assertEqual("Cannot rollback to 2.12.0, current Marqo version is 2.13.0", str(e.exception))

    def test_rollback_should_fail_when_schemas_are_changed(self):
        self.index_management.bootstrap_vespa()

        self.index_management.create_index(self.unstructured_marqo_index_request())

        with mock.patch.object(version, 'get_version', return_value='2.10.0'):
            with self.assertRaisesStrict(ApplicationRollbackError) as e:
                self.index_management.rollback_vespa()
            self.assertEqual("Aborting rollback. Reason: Indexes have been added or removed since last backup.", str(e.exception))

    def test_rollback_should_fail_when_nodes_are_changed(self):
        self.index_management.bootstrap_vespa()

        # Change the container nodes to add a JVM setting
        application = self.index_management._get_vespa_application()
        container_nodes_element = application._service_xml._ensure_only_one('container/nodes')
        # add <jvm options="-Xms32M -Xmx128M"/> to container/nodes
        ET.SubElement(container_nodes_element, 'jvm', {'options': '-Xms32M -Xmx128M'})
        application._store.save_file(application._service_xml.to_xml(), application._SERVICES_XML_FILE)
        application._deploy()

        with mock.patch.object(version, 'get_version', return_value='2.10.0'):
            with self.assertRaisesStrict(ApplicationRollbackError) as e:
                self.index_management.rollback_vespa()
            self.assertEqual("Aborting rollback. Reason: Vector store config has been changed since the last backup.", str(e.exception))

    def test_rollback_should_fail_when_admin_config_is_changed(self):
        self.index_management.bootstrap_vespa()

        application = self.index_management._get_vespa_application()
        root_element = application._service_xml._root
        # add <admin version="2.0"><adminserver hostalias="node1" /></admin> to root element
        admin_element = ET.SubElement(root_element, 'admin', {'version': '2.0'})
        ET.SubElement(admin_element, 'adminserver', {'hostalias': 'node1'})
        application._store.save_file(application._service_xml.to_xml(), application._SERVICES_XML_FILE)
        application._deploy()

        with mock.patch.object(version, 'get_version', return_value='2.10.0'):
            with self.assertRaisesStrict(ApplicationRollbackError) as e:
                self.index_management.rollback_vespa()
            self.assertEqual("Aborting rollback. Reason: Vector store config has been changed since the last backup.",
                             str(e.exception))

    def test_index_operation_methods_should_raise_error_if_index_operation_is_disabled(self):
        # Create an index management instance with index operation disabled (by default)
        self.index_management = IndexManagement(self.vespa_client, zookeeper_client=None)
        index_request_1 = self.structured_marqo_index_request(
            fields=[FieldRequest(name='title', type=FieldType.Text)],
            tensor_fields=['title']
        )
        index_request_2 = self.unstructured_marqo_index_request()

        with self.assertRaisesStrict(InternalError):
            self.index_management.create_index(index_request_1)

        with self.assertRaisesStrict(InternalError):
            self.index_management.batch_create_indexes([index_request_1, index_request_2])

        with self.assertRaisesStrict(InternalError):
            self.index_management.delete_index_by_name(index_request_1.name)

        with self.assertRaisesStrict(InternalError):
            self.index_management.batch_delete_indexes_by_name([index_request_1.name, index_request_2.name])

    def test_index_operation_methods_should_raise_error_if_marqo_is_not_bootstrapped(self):
        index_request_1 = self.structured_marqo_index_request(
            fields=[FieldRequest(name='title', type=FieldType.Text)],
            tensor_fields=['title']
        )
        index_request_2 = self.unstructured_marqo_index_request()

        with self.assertRaisesStrict(ApplicationNotInitializedError):
            self.index_management.create_index(index_request_1)

        with self.assertRaisesStrict(ApplicationNotInitializedError):
            self.index_management.batch_create_indexes([index_request_1, index_request_2])

        with self.assertRaisesStrict(ApplicationNotInitializedError):
            self.index_management.delete_index_by_name(index_request_1.name)

        with self.assertRaisesStrict(ApplicationNotInitializedError):
            self.index_management.batch_delete_indexes_by_name([index_request_1.name, index_request_2.name])

    def test_create_and_delete_index_should_succeed(self):
        # merge batch create and delete happy path to save some testing time
        request = self.unstructured_marqo_index_request(model=Model(name='hf/e5-small'))
        schema, index = vespa_schema_factory(request).generate_schema()
        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        app = self.vespa_client.download_application()
        self._assert_index_is_present(app, index, schema)

        self.index_management.delete_index_by_name(index.name)

        app = self.vespa_client.download_application()
        self._assert_index_is_not_present(app, index.name, index.schema_name)

    def test_update_index_should_succeed(self):
        request = self.unstructured_marqo_index_request(model=Model(name='hf/e5-small'))
        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        # update this index
        semi_structured_marqo_index = cast(SemiStructuredMarqoIndex, self.index_management.get_index(request.name))
        semi_structured_marqo_index.tensor_fields.append(TensorField(
            name='title', chunk_field_name='marqo__chunks_title', embeddings_field_name='marqo__embeddings_title'))
        vespa_schema = cast(SemiStructuredVespaSchema, vespa_schema_factory(request))
        new_schema = vespa_schema.generate_vespa_schema(semi_structured_marqo_index)
        self.index_management.update_index(semi_structured_marqo_index)

        # verify if the index is updated
        app = self.vespa_client.download_application()
        self._assert_index_is_present(app, semi_structured_marqo_index, new_schema, expected_version=2)

    def test_update_index_should_fail_under_race_condition(self):
        request = self.unstructured_marqo_index_request(model=Model(name='hf/e5-small'))
        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        semi_structured_marqo_index = cast(SemiStructuredMarqoIndex, self.index_management.get_index(request.name))

        change1 = semi_structured_marqo_index.copy(deep=True)
        change1.tensor_fields.append(TensorField(
            name='title', chunk_field_name='marqo__chunks_title', embeddings_field_name='marqo__embeddings_title'))
        self.index_management.update_index(change1)

        change2 = semi_structured_marqo_index.copy(deep=True)
        change2.tensor_fields.append(TensorField(
            name='desc', chunk_field_name='marqo__chunks_desc', embeddings_field_name='marqo__embeddings_desc'))

        with self.assertRaisesStrict(OperationConflictError) as err:
            self.index_management.update_index(change2)

        self.assertIn("Current version is 2, and cannot be upgraded to target version 2. "
                      "Some other request might have changed the index. Please try again.", str(err.exception))

    def test_update_index_should_fail_if_index_does_not_exist(self):
        self.index_management.bootstrap_vespa()

        request = self.unstructured_marqo_index_request(model=Model(name='hf/e5-small'))
        _, index = vespa_schema_factory(request).generate_schema()

        with self.assertRaisesStrict(IndexNotFoundError):
            self.index_management.update_index(index)

    def test_update_index_should_fail_for_wrong_index_types(self):
        self.index_management.bootstrap_vespa()

        for request in [
            # legacy unstructured index cannot be updated
            self.unstructured_marqo_index_request(model=Model(name='hf/e5-small'), marqo_version='2.12.0'),
            # structured index cannot be updated
            self.structured_marqo_index_request(
                fields=[FieldRequest(name='title', type=FieldType.Text)],
                tensor_fields=['title']
            )
        ]:
            with self.subTest(f'request_type={type(request)}'):
                _, index = vespa_schema_factory(request).generate_schema()

                with self.assertRaisesStrict(InternalError):
                    self.index_management.update_index(index)

    def test_create_index_should_fail_if_index_already_exists(self):
        request = self.unstructured_marqo_index_request(name="test-index")
        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        with self.assertRaisesStrict(IndexExistsError):
            self.index_management.create_index(request)

    def test_delete_index_should_fail_when_index_is_not_found(self):
        self.index_management.bootstrap_vespa()

        with self.assertRaisesStrict(IndexNotFoundError):
            self.index_management.delete_index_by_name('index-does-not-exist')

    def test_batch_create_and_delete_index_should_succeed(self):
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

        all_indexes = {index.name: index for index in self.index_management.get_all_indexes()}
        self.assertEqual(2, len(all_indexes))
        exclude_fields = {'model', 'version'}
        for index in [index1, index2]:
            self.assertEqual(all_indexes[index.name].dict(exclude=exclude_fields), index.dict(exclude=exclude_fields))

        self.index_management.batch_delete_indexes_by_name([request1.name, request2.name])

        app = self.vespa_client.download_application()
        self._assert_index_is_not_present(app, index1.name, index1.schema_name)
        self._assert_index_is_not_present(app, index2.name, index2.schema_name)

        self.assertEqual(0, len(self.index_management.get_all_indexes()))

    def test_batch_create_index_should_fail_atomically(self):
        request = self.unstructured_marqo_index_request(name="index1")
        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        request2 = self.unstructured_marqo_index_request(name="index2")
        _, index2 = vespa_schema_factory(request2).generate_schema()

        with self.assertRaisesStrict(IndexExistsError):
            self.index_management.batch_create_indexes([request2, request])

        app = self.vespa_client.download_application()
        self._assert_index_is_not_present(app, index2.name, index2.schema_name)

    def test_batch_delete_index_should_fail_atomically(self):
        request = self.unstructured_marqo_index_request(name="index1")
        schema, index1 = vespa_schema_factory(request).generate_schema()

        self.index_management.bootstrap_vespa()
        self.index_management.create_index(request)

        request2 = self.unstructured_marqo_index_request(name="index2")
        _, index2 = vespa_schema_factory(request2).generate_schema()

        with self.assertRaisesStrict(IndexNotFoundError):
            self.index_management.batch_delete_indexes_by_name([request.name, request2.name])

        app = self.vespa_client.download_application()
        self._assert_index_is_present(app, index1, schema)
        self._assert_index_is_not_present(app, index2.name, index2.schema_name)

    def test_concurrent_updates_is_prevented_by_distributed_locking(self):
        def worker1():
            request = self.unstructured_marqo_index_request(name="index1")
            self.index_management.create_index(request)

        def worker2():
            with self.assertRaisesStrict(OperationConflictError) as e:
                request = self.unstructured_marqo_index_request(name="index2")
                self.index_management.create_index(request)
            self.assertEqual("Another index creation/deletion operation is in progress.", str(e.exception))

        self.index_management.bootstrap_vespa()
        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)

        thread1.start()
        time.sleep(1)
        thread2.start()
        thread1.join()
        thread2.join()


    @pytest.mark.skip(reason="This test case is just used to verify the optimistic locking mechanism works")
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

        content_base_url1, prepare_url1 = self.vespa_client.create_deployment_session()
        store1 = ApplicationPackageDeploymentSessionStore(content_base_url1, self.vespa_client)
        app1 = VespaApplicationPackage(store1)

        content_base_url2, prepare_url2 = self.vespa_client.create_deployment_session()
        store2 = ApplicationPackageDeploymentSessionStore(content_base_url2, self.vespa_client)
        app2 = VespaApplicationPackage(store2)

        app1.batch_add_index_setting_and_schema([(schema1, index1)])
        app2.batch_add_index_setting_and_schema([(schema2, index2)])

        prepare_res1 = self.vespa_client.prepare(prepare_url1)
        self.vespa_client.activate(prepare_res1['activate'])

        prepare_res2 = self.vespa_client.prepare(prepare_url2)
        # this should fail due to optimistic locking
        with self.assertRaisesStrict(VespaActivationConflictError):
            self.vespa_client.activate(prepare_res2['activate'])

    def _assert_file_exists(self, *file_paths: str):
        self.assertTrue(os.path.exists(os.path.join(*file_paths)), f'File {"/".join(file_paths[1:])} does not exist')

    def _assert_file_does_not_exist(self, *file_paths: str):
        self.assertFalse(os.path.exists(os.path.join(*file_paths)),f'File {"/".join(file_paths[1:])} exists')

    def _assert_files_equal(self, path1: str, path2: str):
        self.assertTrue(filecmp.cmp(path1, path2),
                        f'Expect file {path1} and {path2} to have same content, but they differ')

    def _assert_files_not_equal(self, path1: str, path2: str):
        self.assertFalse(filecmp.cmp(path1, path2),
                         f'Expect file {path1} and {path2} to have different content, but they are the same')

    def _assert_index_is_present(self, app, expected_index, expected_schema, expected_version=1):
        # assert index setting exists and equals to expected value
        saved_index = self.index_management.get_index(expected_index.name)
        exclude_fields = {'model', 'version'}
        self.assertEqual(saved_index.dict(exclude=exclude_fields), expected_index.dict(exclude=exclude_fields))
        self.assertEqual(saved_index.version, expected_version)

        # asser that the prefixes are set correctly
        model_properties = get_model_properties_from_registry(saved_index.model.name)
        if 'text_chunk_prefix' in model_properties:
            self.assertEqual(saved_index.model.text_chunk_prefix, model_properties['text_chunk_prefix'])
        if 'text_query_prefix' in model_properties:
            self.assertEqual(saved_index.model.text_query_prefix, model_properties['text_query_prefix'])

        # assert schema file exists and has expected value
        schema_name = expected_index.schema_name
        self._assert_file_exists(app, 'schemas', f'{schema_name}.sd')
        with open(os.path.join(app, 'schemas', f'{schema_name}.sd')) as f:
            self.assertEqual(f.read(), expected_schema)
        doc = ET.parse(os.path.join(app, 'services.xml')).getroot().find(f'content/documents/document[@type="{schema_name}"]')
        self.assertIsNotNone(doc)

    def _assert_index_is_not_present(self, app, index_name, schema_name):
        with self.assertRaisesStrict(IndexNotFoundError):
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
        _, index = vespa_schema_factory(self.unstructured_marqo_index_request(
            name="existing_index", marqo_version='2.10.0')).generate_schema()

        app_root_path = os.path.join(self._test_dir, 'existing_vespa_app')
        self._add_schema_removal_override(app_root_path)
        self.vespa_client.deploy_application(app_root_path)
        self.vespa_client.wait_for_application_convergence()

        self._save_index_settings_to_vespa(index)
        self._save_marqo_version_to_vespa('2.10.0')

        return index

    def _add_schema_removal_override(self, app_root_path: str):
        content = textwrap.dedent(
            f'''
            <validation-overrides>
                 <allow until='{datetime.utcnow().strftime('%Y-%m-%d')}'>schema-removal</allow>
            </validation-overrides>
            '''
        ).strip()
        with open(os.path.join(app_root_path, 'validation-overrides.xml'), 'w') as f:
            f.write(content)

    def _save_marqo_version_to_vespa(self, version: str) -> None:
        self.vespa_client.feed_document(
            VespaDocument(
                id=self.index_management._MARQO_CONFIG_DOC_ID,
                fields={'settings': MarqoConfig(version=version).json()}
            ),
            schema=self.index_management._MARQO_SETTINGS_SCHEMA_NAME
        )

    def _save_index_settings_to_vespa(self, marqo_index: MarqoIndex) -> None:
        self.vespa_client.feed_document(
            VespaDocument(
                id=marqo_index.name,
                fields={'index_name': marqo_index.name, 'settings': marqo_index.json()}
            ),
            schema=self.index_management._MARQO_SETTINGS_SCHEMA_NAME
        )
