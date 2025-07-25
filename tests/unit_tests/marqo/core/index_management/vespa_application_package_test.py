import os
from unittest.mock import MagicMock

from marqo.core.index_management.vespa_application_package import ServicesXml, VespaApplicationPackage
from tests.unit_tests.marqo_test import MarqoTestCase



class TestSemiStructuredPartialUpdate(MarqoTestCase):
    def setUp(self):
        self.base_xml_str = """<?xml version="1.0" encoding="utf-8" standalone="no"?>
<services xmlns:deploy="vespa" xmlns:preprocess="properties" version="1.0">
        
    <container id="default" version="1.0">
                
        <document-api/>
        <search>
            <chain id="marqo" inherits="vespa">
                <searcher bundle="marqo-custom-searchers" id="ai.marqo.search.HybridSearcher"/>
            </chain>
        </search>
        <nodes>
                        
            <node hostalias="node1"/>
                    
        </nodes>
            
        <handler bundle="marqo-custom-searchers" id="ai.marqo.index.IndexSettingRequestHandler">
            <binding>http://*/index-settings/*</binding>
            <binding>http://*/index-settings</binding>
        </handler>
        <component bundle="marqo-custom-searchers" id="ai.marqo.index.IndexSettings">
            <config name="ai.marqo.index.index-settings">
                <indexSettingsFile>marqo_index_settings.json</indexSettingsFile>
                <indexSettingsHistoryFile>marqo_index_settings_history.json</indexSettingsHistoryFile>
            </config>
        </component>
    </container>
        
    <content id="content_default" version="1.0">
                
        <redundancy>2</redundancy>
                
        <documents>
                        
            <document mode="index" type="test_vespa_client"/>
                    
            <document mode="index" type="a3b6363b700334201aaa71448ab76e341"/>
        </documents>
                
        <nodes>
                        
            <node distribution-key="0" hostalias="node1"/>
                    
        </nodes>
            
    </content>
    
</services>
"""
        self.vespa_application_store = MagicMock()
        self.vespa_application_store.read_text_file.side_effect = lambda *args: self.base_xml_str if args and 'services.xml' in args[0] else None
        self.vespa_application_package = VespaApplicationPackage(self.vespa_application_store)


    def test_compare_element_no_changes(self):
        """Test that compare_element returns None when there are no changes."""

        other = ServicesXml(self.base_xml_str)
        result = self.vespa_application_package._service_xml.compare_element(other, "content/documents")
        self.assertTrue(result)

    def test_compare_content_documents_pagination_document_is_ignored(self):
        """Test that compare_element returns True when there are no changes except for pagination schema."""
        self.vespa_application_package._configure_pagination_schema(MagicMock())
        other = ServicesXml(self.base_xml_str)
        result = self.vespa_application_package._service_xml.compare_element(other, "content/documents")
        self.assertTrue(result)

    def test_compare_content_documents_document_changes_other_than_pagination_return_false(self):
        """Test that compare_element returns False when there are changes other than pagination."""
        other_xml_str = self.base_xml_str.replace(
            '<document mode="index" type="test_vespa_client"/>',
            '<document mode="index" type="test_vespa_client_changed"/>'
        )
        other = ServicesXml(other_xml_str)
        result = self.vespa_application_package._service_xml.compare_element(other, "content/documents")
        self.assertFalse(result)

    def test_configure_pagination_schema(self):
        """Test that _configure_pagination_schema correctly adds the pagination schema."""
        self.vespa_application_package._configure_pagination_schema(MagicMock())
        expected_new_doc = '<document type="marqo__pagination" mode="index" selection="marqo__pagination.updated_at &gt; now() - 1800" />'
        self.assertIn(expected_new_doc, str(self.vespa_application_package._service_xml))
        expected_new_documents = '<documents garbage-collection="true" garbage-collection-interval="1800">'
        self.assertIn(expected_new_documents, str(self.vespa_application_package._service_xml))

    def test_configure_pagination_schema_garbage_collection_interval_overrides_work(self):
        """Test that _configure_pagination_schema correctly sets the garbage collection interval."""
        os.environ["PAGINATION_GARBAGE_COLLECTION_INTERVAL"] = "300"
        os.environ["PAGINATION_TTL"] = "600"
        self.vespa_application_package._configure_pagination_schema(MagicMock())
        expected_new_doc = '<document type="marqo__pagination" mode="index" selection="marqo__pagination.updated_at &gt; now() - 600" />'
        self.assertIn(expected_new_doc, str(self.vespa_application_package._service_xml))
        expected_new_documents = '<documents garbage-collection="true" garbage-collection-interval="300">'
        self.assertIn(expected_new_documents, str(self.vespa_application_package._service_xml))


