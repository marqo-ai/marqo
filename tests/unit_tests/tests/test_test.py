from tests.unit_tests.tests.marqo_test import MarqoTestCase


class TestDummy(MarqoTestCase):
    def test_always_passes(self):
        self.assertTrue(True)

