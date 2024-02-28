import os
import shutil
import uuid
from unittest import mock

from marqo import version
from marqo.core.exceptions import IndexExistsError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import VespaDocument
from tests.marqo_test import MarqoTestCase


class TestIndexManagement(MarqoTestCase):

    def setUp(self):
        self.index_management = IndexManagement(self.vespa_client)

    def test_bootstrap_vespa_doesNotExist_successful(self):
        settings_schema_name = 'a' + str(uuid.uuid4()).replace('-', '')
        with mock.patch.object(IndexManagement, '_MARQO_SETTINGS_SCHEMA_NAME', settings_schema_name):
            self.assertTrue(self.index_management.bootstrap_vespa())

            # Verify settings schema exists
            try:
                self.vespa_client.feed_document(
                    VespaDocument(
                        id='1',
                        fields={}
                    ),
                    schema=settings_schema_name
                )
            except VespaStatusError as e:
                if e.status_code == 400:
                    self.fail('Settings schema does not exist')
                else:
                    raise e

            # Verify default query profile exists
            app = self.vespa_client.download_application()
            query_profile_exists = os.path.exists(
                os.path.join(app, 'search/query-profiles', 'default.xml')
            )
            self.assertTrue(query_profile_exists, 'Default query profile does not exist')

    def test_bootstrap_vespa_exists_skips(self):
        settings_schema_name = 'a' + str(uuid.uuid4()).replace('-', '')
        with mock.patch.object(IndexManagement, '_MARQO_SETTINGS_SCHEMA_NAME', settings_schema_name):
            self.assertTrue(self.index_management.bootstrap_vespa())

            import httpx

            def modified_post(*args, **kwargs):
                self.assertFalse(
                    'prepareandactivate' in args[0],
                    'Settings schema deployment must be skipped'
                )
                return httpx.post(*args, **kwargs)

            with mock.patch.object(httpx.Client, 'post', side_effect=modified_post) as mock_post:
                self.assertFalse(self.index_management.bootstrap_vespa())
                # Sanity check that we're patching the right method
                self.assertTrue(mock_post.called)

    def test_boostrap_vespa_v2Exists_skips(self):
        """
        bootstrap_vespa skips when Vespa has been configured with Marqo 2.0.x
        """
        # Marqo 2.0.x configuration is detected by presence of settings schema, but absence default query profile
        settings_schema_name = 'a' + str(uuid.uuid4()).replace('-', '')
        with mock.patch.object(IndexManagement, '_MARQO_SETTINGS_SCHEMA_NAME', settings_schema_name):
            app = self.vespa_client.download_application()

            # Clean any query profiles that may exist
            shutil.rmtree(os.path.join(app, 'search'), ignore_errors=True)

            self.index_management._add_marqo_settings_schema(app)
            self.vespa_client.deploy_application(app)
            self.vespa_client.wait_for_application_convergence()

            self.assertFalse(
                self.index_management.bootstrap_vespa(),
                'bootstrap_vespa should skip when Marqo 2.0.x configuration is detected'
            )

    def test_bootstrap_vespa_partialConfig_successful(self):
        """
        bootstrap_vespa succeeds when Vespa has been partially configured and recovers to a consistent state
        """
        settings_schema_name = 'a' + str(uuid.uuid4()).replace('-', '')
        with mock.patch.object(IndexManagement, '_MARQO_SETTINGS_SCHEMA_NAME', settings_schema_name):
            self.assertTrue(self.index_management.bootstrap_vespa())

            # Delete marqo config to simulate partial configuration for 2.1+
            self.vespa_client.delete_document(
                schema=settings_schema_name,
                id=IndexManagement._MARQO_CONFIG_DOC_ID
            )

            self.assertTrue(self.index_management.bootstrap_vespa(), 'bootstrap_vespa should not skip')
            # Verify config has been saved
            self.assertEqual(version.get_version(), self.index_management.get_marqo_version())

    def test_create_index_settingsSchemaDoesNotExist_successful(self):
        """
        A new index is created successfully when the settings schema does not exist
        """
        index_name = 'a' + str(uuid.uuid4()).replace('-', '')
        settings_schema_name = 'a' + str(uuid.uuid4()).replace('-', '')
        marqo_index_request = self.structured_marqo_index_request(
            name=index_name,
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text),
                FieldRequest(name='description', type=FieldType.Text),
                FieldRequest(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier])
            ],
            tensor_fields=['title', 'description']
        )

        with mock.patch.object(IndexManagement, '_MARQO_SETTINGS_SCHEMA_NAME', settings_schema_name):
            self.index_management.create_index(marqo_index_request)

            # Inserting a document into the new schema to verify it exists
            self.vespa_client.feed_document(
                VespaDocument(
                    id='1',
                    fields={}
                ),
                schema=index_name
            )

            # Verify settings have been saved
            settings_json = self.pyvespa_client.get_data(
                schema=IndexManagement._MARQO_SETTINGS_SCHEMA_NAME,
                data_id=index_name
            ).json['fields']['settings']

            # Generate Marqo Index to compare
            _, marqo_index_request = vespa_schema_factory(marqo_index_request).generate_schema()

            self.assertEqual(settings_json, marqo_index_request.json())

    def test_create_index_settingsSchemaExists_successful(self):
        """
        A new index is created successfully when the settings schema already exists
        """
        index_name_1 = 'a' + str(uuid.uuid4()).replace('-', '')
        index_name_2 = 'a' + str(uuid.uuid4()).replace('-', '')
        marqo_index_request = self.structured_marqo_index_request(
            name=index_name_1,
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text),
                FieldRequest(name='description', type=FieldType.Text),
                FieldRequest(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier])
            ],
            tensor_fields=['title', 'description']
        )
        self.index_management.create_index(marqo_index_request)

        # Create with a different name now that we know settings schema exists
        marqo_index_request_2 = marqo_index_request.copy(update={'name': index_name_2})

        self.index_management.create_index(marqo_index_request_2)

        # Inserting a document into the new schema to verify it exists
        self.vespa_client.feed_batch(
            [
                VespaDocument(
                    id='1',
                    fields={}
                )
            ],
            schema=index_name_2
        )

        # Verify settings have been saved
        settings_json = self.pyvespa_client.get_data(
            schema=IndexManagement._MARQO_SETTINGS_SCHEMA_NAME,
            data_id=index_name_2
        ).json['fields']['settings']

        # Generate Marqo Index to compare
        _, marqo_index = vespa_schema_factory(marqo_index_request_2).generate_schema()

        self.assertEqual(settings_json, marqo_index.json())

    def test_create_index_indexExists_fails(self):
        """
        An error is raised when creating an index that already exists
        """
        index_name = 'a' + str(uuid.uuid4()).replace('-', '')
        marqo_index_request = self.structured_marqo_index_request(
            name=index_name,
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
            ],
            tensor_fields=[]
        )

        self.index_management.create_index(marqo_index_request)

        with self.assertRaises(IndexExistsError):
            self.index_management.create_index(marqo_index_request)

    def test_get_marqo_version_successful(self):
        """
        get_marqo_version returns current version
        """
        settings_schema_name = 'a' + str(uuid.uuid4()).replace('-', '')
        with mock.patch.object(IndexManagement, '_MARQO_SETTINGS_SCHEMA_NAME', settings_schema_name):
            self.index_management.bootstrap_vespa()

            self.assertEqual(version.get_version(), self.index_management.get_marqo_version())

    def test_get_marqo_version_v20_successful(self):
        """
        get_marqo_version returns 2.0 when Vespa has been configured with Marqo 2.0.x
        """
        settings_schema_name = 'a' + str(uuid.uuid4()).replace('-', '')
        with mock.patch.object(IndexManagement, '_MARQO_SETTINGS_SCHEMA_NAME', settings_schema_name):
            self.index_management.bootstrap_vespa()

            # Delete Marqo config to simulate 2.0
            self.vespa_client.delete_document(
                schema=settings_schema_name,
                id=IndexManagement._MARQO_CONFIG_DOC_ID
            )

            self.assertEqual(self.index_management.get_marqo_version(), '2.0')
