from unittest import mock

from marqo.core.models import MarqoQuery
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.exceptions import InvalidArgumentError
from marqo.settings.settings import Settings
from tests.unit_tests.marqo_test import MarqoTestCase


class TestSemiStructuredInFilter(MarqoTestCase):
    """Unit tests for _id IN filter on semi-structured indexes."""

    def setUp(self):
        self.vespa_index = SemiStructuredVespaIndex(self.semi_structured_marqo_index(
            name='test_index',
            lexical_field_names=['title'],
            tensor_field_names=['title'],
        ))

    def _get_filter(self, filter_string: str) -> str:
        marqo_query = MarqoQuery(
            index_name=self.vespa_index._marqo_index.name,
            limit=10,
            filter=filter_string,
            score_modifiers=[],
            expose_facets=False
        )
        return self.vespa_index._get_filter_term(marqo_query)

    def test_id_in_basic(self):
        """Basic _id IN filter generates correct Vespa YQL."""
        result = self._get_filter('_id IN (doc1, doc2, doc3)')
        self.assertEqual(result, 'marqo__id in ("doc1", "doc2", "doc3")')

    def test_id_in_single_value(self):
        """Single value IN list has no trailing comma."""
        result = self._get_filter('_id IN (doc1)')
        self.assertEqual(result, 'marqo__id in ("doc1")')

    def test_id_in_escaped_quotes(self):
        r"""IDs with double quotes are escaped with backslash."""
        result = self._get_filter(r'_id IN (he\"llo, wor\"ld)')
        self.assertIn(r'he\"llo', result)
        self.assertIn(r'wor\"ld', result)
        self.assertTrue(result.startswith('marqo__id in ('))

    def test_id_in_escaped_backslash(self):
        r"""IDs with backslashes are escaped."""
        result = self._get_filter(r'_id IN (back\\slash)')
        self.assertIn(r'back\\slash', result)

    def test_not_id_in(self):
        """NOT _id IN generates negated Vespa expression."""
        result = self._get_filter('NOT _id IN (doc1, doc2)')
        self.assertEqual(result, '!(marqo__id in ("doc1", "doc2"))')

    def test_id_in_combined_with_and(self):
        """_id IN combined with AND equality filter."""
        result = self._get_filter('_id IN (doc1, doc2) AND title:hello')
        self.assertEqual(
            result,
            '(marqo__id in ("doc1", "doc2") AND '
            '((marqo__short_string_fields contains sameElement(key contains "title", value contains "hello"))))'
        )

    def test_id_in_combined_with_or(self):
        """_id IN combined with OR equality filter."""
        result = self._get_filter('_id IN (doc1) OR title:hello')
        self.assertEqual(
            result,
            '(marqo__id in ("doc1") OR '
            '((marqo__short_string_fields contains sameElement(key contains "title", value contains "hello"))))'
        )

    def test_id_in_empty_returns_empty_string_value(self):
        """_id IN () parses as single empty string value — returns 0 hits in practice."""
        result = self._get_filter('_id IN ()')
        self.assertEqual(result, 'marqo__id in ("")')

    def test_non_id_field_in_raises_error(self):
        """IN on non-_id fields raises InvalidArgumentError."""
        with self.assertRaises(InvalidArgumentError) as cm:
            self._get_filter('color IN (red, blue)')
        self.assertIn("only supported for the '_id' field", str(cm.exception))

    def test_non_id_field_in_raises_error_numeric(self):
        """IN on numeric-looking non-_id fields raises InvalidArgumentError."""
        with self.assertRaises(InvalidArgumentError) as cm:
            self._get_filter('count IN (1, 2, 3)')
        self.assertIn("only supported for the '_id' field", str(cm.exception))

    def test_not_non_id_field_in_raises_error(self):
        """NOT IN on non-_id fields also raises InvalidArgumentError."""
        with self.assertRaises(InvalidArgumentError) as cm:
            self._get_filter('NOT color IN (red, blue)')
        self.assertIn("only supported for the '_id' field", str(cm.exception))

    def test_id_in_many_values(self):
        """_id IN with many values generates correct YQL."""
        ids = [f'id_{i}' for i in range(100)]
        filter_str = '_id IN (' + ', '.join(ids) + ')'
        result = self._get_filter(filter_str)
        self.assertTrue(result.startswith('marqo__id in ('))
        for id_ in ids:
            self.assertIn(f'"{id_}"', result)

    def test_id_in_exceeds_max_limit_raises_error(self):
        """_id IN exceeding MARQO_MAX_IN_FILTER_IDS raises InvalidArgumentError."""
        max_ids = 3
        ids = [f'id_{i}' for i in range(max_ids + 1)]
        filter_str = '_id IN (' + ', '.join(ids) + ')'

        with mock.patch("marqo.settings.settings._settings", Settings(marqo_max_in_filter_ids=max_ids)):
            with self.assertRaises(InvalidArgumentError) as cm:
                self._get_filter(filter_str)

            self.assertIn("MARQO_MAX_IN_FILTER_IDS", str(cm.exception))
            self.assertIn(str(max_ids), str(cm.exception))

    def test_id_in_at_max_limit_succeeds(self):
        """_id IN with exactly MARQO_MAX_IN_FILTER_IDS values succeeds."""
        max_ids = 5
        ids = [f'id_{i}' for i in range(max_ids)]
        filter_str = '_id IN (' + ', '.join(ids) + ')'

        with mock.patch("marqo.settings.settings._settings", Settings(marqo_max_in_filter_ids=max_ids)):
            result = self._get_filter(filter_str)
            self.assertTrue(result.startswith('marqo__id in ('))

    def test_non_id_in_exceeds_max_limit_raises_field_error_not_limit_error(self):
        """Non-_id IN raises field error even if limit is also exceeded."""
        with mock.patch("marqo.settings.settings._settings", Settings(marqo_max_in_filter_ids=1)):
            with self.assertRaises(InvalidArgumentError) as cm:
                self._get_filter('color IN (red, blue, green)')

            self.assertIn("only supported for the '_id' field", str(cm.exception))
            self.assertNotIn("MARQO_MAX_IN_FILTER_IDS", str(cm.exception))

    def test_non_id_in_exceeds_default_limit_raises_field_error(self):
        """Non-_id IN with 10,001 values raises field error, not limit error."""
        ids = [f'val_{i}' for i in range(10001)]
        filter_str = 'color IN (' + ', '.join(ids) + ')'

        with self.assertRaises(InvalidArgumentError) as cm:
            self._get_filter(filter_str)

        self.assertIn("only supported for the '_id' field", str(cm.exception))
        self.assertNotIn("MARQO_MAX_IN_FILTER_IDS", str(cm.exception))
