from unittest import TestCase

from marqo_common.version import __version__ as common_version, get_version as common_get_version
from marqo.version import __version__, get_version


class TestVersion(TestCase):
    def test_version_format(self):
        """Version string should follow semver format."""
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")

    def test_get_version_returns_version(self):
        self.assertEqual(get_version(), __version__)

    def test_version_matches_common(self):
        """Component version should be sourced from marqo_common."""
        self.assertEqual(__version__, common_version)

    def test_get_version_matches_common(self):
        self.assertEqual(get_version(), common_get_version())
