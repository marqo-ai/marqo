import re
import textwrap
import unittest
from string import Template

import pytest

from marqo.core.exceptions import InternalError
from marqo.core.index_management.vespa_application_package import ServicesXml


@pytest.mark.unnittest
class TestIndexSettingStore(unittest.TestCase):

    _TEMPLATE = Template(textwrap.dedent("""<?xml version="1.0" encoding="utf-8" ?>
            <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
                <content id="content_default" version="1.0">
                    <documents>$documents</documents>
                </content>
            </services>
        """))

    def test_compare_element_should_return_true_when_equals_semantically(self):

        xml1 = self._TEMPLATE.substitute(documents="""
                    <document type="marqo__settings" mode="index"/>
                    <document type="marqo__existing_00index" mode="index"/>
                """)

        # we changed the order of the document sub elements and its attributes in documents
        # This is still the same semantically
        xml2 = self._TEMPLATE.substitute(documents="""
                    <document type="marqo__existing_00index" mode="index"/>
                    <document mode="index" type="marqo__settings"></document>
                """)

        self.assertTrue(ServicesXml(xml1).compare_element(ServicesXml(xml2), 'content/documents'))
        self.assertTrue(ServicesXml(xml1).compare_element(ServicesXml(xml2), 'content/documents/document'))

    def test_compare_element_should_return_false_when_not_equal_semantically(self):
        xml1 = self._TEMPLATE.substitute(documents="""
                    <document type="marqo__existing_00index" mode="index"/>
                """)
        # we changed the order of the document sub elements and its attributes in documents
        # This is still the same semantically
        for test_case in [
            """<document type="marqo__existing_00index_01" mode="index"/>""",  # different document
            """""",  # no documents
            # an extra documents
            """<document type="marqo__existing_00index" mode="index"/><document mode="index" type="marqo__settings"/>""",
        ]:
            with self.subTest():
                xml2 = self._TEMPLATE.substitute(documents=test_case)
                self.assertFalse(ServicesXml(xml1).compare_element(ServicesXml(xml2), 'content/documents'))
                self.assertFalse(ServicesXml(xml1).compare_element(ServicesXml(xml2), 'content/documents/document'))

    def test_compare_element_should_detect_any_difference_in_multiple_elements(self):
        xml1 = """<?xml version="1.0" encoding="utf-8" ?>
                    <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
                        <container id="default" version="1.0">
                            <nodes><node hostalias="node1"/></nodes>
                        </container>
                        <content id="content_default" version="1.0">
                            <documents><document type="marqo__existing_00index" mode="index"/></documents>
                            <nodes><node hostalias="node2"/></nodes>
                        </content>
                    </services>
                """
        # added node3 in content element
        xml2 = """<?xml version="1.0" encoding="utf-8" ?>
                    <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
                        <container id="default" version="1.0">
                            <nodes><node hostalias="node1"/></nodes>
                        </container>
                        <content id="content_default" version="1.0">
                            <documents><document type="marqo__existing_00index" mode="index"/></documents>
                            <nodes><node hostalias="node2"/><node hostalias="node3"/></nodes>
                        </content>
                    </services>
                """

        self.assertFalse(ServicesXml(xml1).compare_element(ServicesXml(xml2), '*/nodes'))

    def test_should_not_have_more_than_one_content_documents_element(self):
        xml = """<?xml version="1.0" encoding="utf-8" ?>
            <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
                <content id="content_default" version="1.0">
                    <documents>
                        <document type="marqo__settings" mode="index"/>
                    </documents>
                </content>
                <content id="another" version="1.0">
                    <documents>
                        <document type="marqo__settings" mode="index"/>
                    </documents>
                </content>
            </services>
        """

        with self.assertRaises(InternalError) as e:
            ServicesXml(xml)
        self.assertEqual('Multiple content/documents elements found in services.xml. Only one is allowed',
                         str(e.exception))

    def test_should_have_one_content_document_element(self):
        xml = """<?xml version="1.0" encoding="utf-8" ?>
                    <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
                        <content id="content_default" version="1.0">    
                        </content>
                    </services>
                """

        with self.assertRaises(InternalError) as e:
            ServicesXml(xml)
        self.assertEqual('No content/documents element found in services.xml', str(e.exception))

    def test_add_schema_should_skip_when_schema_exists(self):
        xml = self._TEMPLATE.substitute(documents="""<document type="marqo__existing_00index" mode="index"/>""")
        service_xml = ServicesXml(xml)
        service_xml.add_schema("marqo__existing_00index")
        self._assertStringsEqualIgnoringWhitespace(xml, service_xml.to_xml())

    def test_add_schema(self):
        xml = self._TEMPLATE.substitute(documents="""""")
        service_xml = ServicesXml(xml)

        expected_xml = self._TEMPLATE.substitute(documents="""<document type="marqo__existing_00index" mode="index"/>""")
        service_xml.add_schema("marqo__existing_00index")

        self._assertStringsEqualIgnoringWhitespace(expected_xml, service_xml.to_xml())

    def test_remove_schema_should_skip_if_not_exist(self):
        xml = self._TEMPLATE.substitute(documents="""<document type="marqo__existing_00index" mode="index"/>""")
        service_xml = ServicesXml(xml)
        service_xml.remove_schema("new_00schema")
        self._assertStringsEqualIgnoringWhitespace(xml, service_xml.to_xml())

    def test_remove_schema(self):
        xml = self._TEMPLATE.substitute(documents="""
            <document type="marqo__existing_00index" mode="index"/>
            <document type="new_00index" mode="index"/>
        """)
        service_xml = ServicesXml(xml)
        service_xml.remove_schema("marqo__existing_00index")
        expected_xml = self._TEMPLATE.substitute(documents="""<document type="new_00index" mode="index"/>""")
        self._assertStringsEqualIgnoringWhitespace(expected_xml, service_xml.to_xml())

    def test_config_components(self):
        xml = """<?xml version="1.0" encoding="utf-8" ?>
                    <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
                        <container id="default" version="1.0">
                            <document-api/>
                            <search/>
                            <component id="ai.marqo.RandomComponent" bundle="marqo-custom-searchers"/>
                            <random-element/>
                            <nodes>
                                <node hostalias="node1"/>
                            </nodes>
                        </container>
                        <content id="content_default" version="1.0">
                            <documents><document type="marqo__existing_00index" mode="index"/></documents>
                        </content>
                    </services>
                """

        service_xml = ServicesXml(xml)
        service_xml.config_components()

        # removed all random element and custom components from container element
        # added searcher, handler and other custom components to container element
        # kept nodes in container as is
        # kept content element as is
        expected_xml = """<?xml version="1.0" encoding="utf-8" ?>
                    <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
                        <container id="default" version="1.0">
                            <document-api/>
                            <search>
                                <chain id="marqo" inherits="vespa">
                                    <searcher id="ai.marqo.search.HybridSearcher" bundle="marqo-custom-searchers"/>
                                </chain>
                            </search>
                            <nodes>
                                <node hostalias="node1"/>
                            </nodes>
                            <handler id="ai.marqo.index.IndexSettingRequestHandler" bundle="marqo-custom-searchers">
                                <binding>http://*/index-settings/*</binding>
                                <binding>http://*/index-settings</binding>
                            </handler>
                            <component id="ai.marqo.index.IndexSettings" bundle="marqo-custom-searchers">
                                <config name="ai.marqo.index.index-settings">
                                    <indexSettingsFile>marqo_index_settings.json</indexSettingsFile>
                                    <indexSettingsHistoryFile>marqo_index_settings_history.json</indexSettingsHistoryFile>
                                </config>
                            </component>
                        </container>
                        <content id="content_default" version="1.0">
                            <documents><document type="marqo__existing_00index" mode="index"/></documents>
                        </content>
                    </services>
                """
        self._assertStringsEqualIgnoringWhitespace(expected_xml, service_xml.to_xml())

    def _assertStringsEqualIgnoringWhitespace(self, s1: str, s2: str):
        """Custom assertion to compare strings ignoring whitespace."""

        def remove_whitespace(s: str) -> str:
            return re.sub(r'\s+', '', s)

        cleaned_s1 = remove_whitespace(s1)
        cleaned_s2 = remove_whitespace(s2)
        self.assertEqual(cleaned_s1, cleaned_s2)