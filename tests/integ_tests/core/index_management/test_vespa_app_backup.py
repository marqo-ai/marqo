import os
import tempfile
import unittest

import pytest

from marqo.core.index_management.vespa_application_package import VespaAppBackup


@pytest.mark.unittest
class TestIndexSettingStore(unittest.TestCase):

    _file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'existing_vespa_app')

    def test_backup_and_read_back(self):
        backup = VespaAppBackup()
        files_to_backup = [['services.xml'], ['search', 'query-profiles', 'default.xml']]
        files_to_remove = [['random-config.json'], ['components', 'marqo-custom-searchers-deploy.jar']]
        for paths in files_to_backup:
            with open(os.path.join(self._file_dir, *paths), 'rb') as fp:
                backup.backup_file(fp.read(), *paths)

        for paths in files_to_remove:
            backup.mark_for_removal(*paths)

        with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
            temp_file.write(backup.to_zip_stream().read())
            temp_file_path = temp_file.name

        with open(temp_file_path, 'rb') as f:
            backup2 = VespaAppBackup(f.read())

        # Verify we can traverse the files marked as removal
        remove_file_list = []
        for paths in backup2.files_to_remove():
            remove_file_list.append(paths)
        self.assertEqual(files_to_remove, remove_file_list)

        # Verify we can traverse the files we backed up with the correct content
        rollback_file_dict = {}
        for path, content in backup2.files_to_rollback():
            rollback_file_dict[path] = content

        self.assertEqual(len(rollback_file_dict), len(files_to_backup))
        for paths in files_to_backup:
            relpath = os.path.join(*paths)
            self.assertIn(relpath, rollback_file_dict)
            with open(os.path.join(self._file_dir, *paths), 'rb') as fp:
                content = fp.read()
                self.assertEqual(content, rollback_file_dict[relpath])

    def test_read_text_file(self):
        backup = VespaAppBackup()
        services_xml_path = 'services.xml'
        with open(os.path.join(self._file_dir, services_xml_path), 'r') as fp:
            service_xml_content = fp.read()
            backup.backup_file(service_xml_content, services_xml_path)

        self.assertEqual(service_xml_content, backup.read_text_file(services_xml_path))
        self.assertIsNone(backup.read_text_file('some.json'))