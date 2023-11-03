from marqo.core.search.search_filter import *
from tests.marqo_test import MarqoTestCase


class TestMarqoFilterStringParser(MarqoTestCase):

    def test_parse_successful(self):
        test_cases = [
            (
                'a:b',
                SearchFilter(root=EqualityTerm('a', 'b', 'a:b')),
                'single term'
            ),
            (
                '(a:b)',
                SearchFilter(root=EqualityTerm('a', 'b', 'a:b')),
                'single term with parentheses'
            ),
            (
                '(((a:n)))',
                SearchFilter(root=EqualityTerm('a', 'b', 'a:b')),
                'single term with extra parentheses'
            ),
            (
                'a:1 AND b:2 OR c:3',
                SearchFilter(
                    root=Or(
                        left=And(
                            left=EqualityTerm('a', '1', 'a:b'),
                            right=EqualityTerm('a', '2', 'a:b')
                        ),
                        right=EqualityTerm('a', 'b', 'a:b')
                    )
                ),
                'simple filter string'
            ),
            (
                '(((a AND b)) OR c)',
                SearchFilter(
                    root=Or(
                        left=And(
                            left=Term('a'),
                            right=Term('b')
                        ),
                        right=Term('c')
                    )
                ),
                'extra parantheses'
            ),
            (
                'a AND (b OR c)',
                SearchFilter(
                    root=And(
                        left=Term('a'),
                        right=Or(
                            left=Term('b'),
                            right=Term('c')
                        )
                    )
                ),
                'or expression'
            ),
            (
                'a AND (b OR (c AND (d OR e))) OR d',
                SearchFilter(
                    root=Or(
                        left=And(
                            left=Term('a'),
                            right=Or(
                                left=Term('b'),
                                right=And(
                                    left=Term('c'),
                                    right=Or(
                                        left=Term('d'),
                                        right=Term('e')
                                    )
                                )
                            )
                        ),
                        right=Term('d')
                    )
                ),
                'nested filter string'
            )
        ]

        for filter_string, expected_filter, msg in test_cases:
            with self.subTest(msg):
                self.assertEqual(expected_filter, MarqoFilterStringParser.parse(filter_string), msg)

    def test_parse_malformedString_fails(self):
        test_cases = [
            ('AND a OR b', 'string starts with operator'),
            ('a AND b (OR c)', 'expression starts with operator'),
            ('a AND b OR', 'string ends with operator'),
            ('a AND (b OR c AND) OR e', 'expression ends with operator'),
            ('a AND b OR OR', 'operator after operator'),
            ('a a', 'term after term'),
            ('(a AND b) b', 'term after expression'),
            ('(a AND b)(c AND d)', 'expression after expression'),
            ('a (c AND d)', 'expression after term'),
            ('a AND b)', 'expression not opened'),
            ('(a AND b', 'expression not closed'),
            ('', 'empty expression'),
            (' ', 'empty expression'),
            ('   ', 'empty expression'),
            ('(', '('),
            (')', ')'),
            ('()', '()'),
            ('a AND (b OR (c AND (d OR e)) OR d', 'imbalanced parentheses, not closed'),
            ('a AND b OR (c AND (d OR e))) OR d', 'imbalanced parentheses, not opened')
        ]

        for filter_string, msg in test_cases:
            with self.subTest(msg):
                with self.assertRaises(FilterStringParsingError,
                                       msg=f"Did not raise error on malformed filter string '{filter_string}'"):
                    MarqoFilterStringParser.parse(filter_string)

    def test_equality(self):
        """
        Test all equality functions
        """
        tree1 = Or(
            left=And(
                left=Term('a'),
                right=Term('b')
            ),
            right=Term('c')
        )
        # Set one left node to None
        tree2 = Or(
            left=And(
                left=None,
                right=Term('b')
            ),
            right=Term('c')
        )
        # Change value of one term from tree1
        tree3 = Or(
            left=And(
                left=Term('a'),
                right=Term('b')
            ),
            right=Term('e')
        )
        # Only change operator types from tree1
        tree4 = And(
            left=Or(
                left=Term('a'),
                right=Term('b')
            ),
            right=Term('c')
        )

        trees = [tree1, tree2, tree3, tree4]
        for i in range(len(trees)):
            self.assertTrue(trees[i] == trees[i], f'Self-equality failed for tree {i}')

        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                self.assertFalse(trees[i] == trees[j], f'Equality returned True for unequal trees {i} and {j}')
