import textwrap
import unittest
from string import Template

import pytest

from marqo.core.index_management.vespa_application_package import ServiceXml


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

        self.assertTrue(ServiceXml(xml1).compare_element(ServiceXml(xml2), 'content/documents'))

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
                self.assertFalse(ServiceXml(xml1).compare_element(ServiceXml(xml2), 'content/documents'))


