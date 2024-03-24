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
                '(NOT a:b)',
                SearchFilter(root=Not(EqualityTerm('a', 'b', 'a:b'))),
                'single modifier term with parentheses'
            ),
            (
                '(((a:n)))',
                SearchFilter(root=EqualityTerm('a', 'n', 'a:n')),
                'single term with extra parentheses'
            ),
            (
                'NOT a:1 AND b:2 OR NOT c:3',
                SearchFilter(
                    root=Or(
                        left=And(
                            left=Not(EqualityTerm('a', '1', 'a:1')),
                            right=EqualityTerm('b', '2', 'b:2')
                        ),
                        right=Not(EqualityTerm('c', '3', 'c:3'))
                    )
                ),
                'simple filter string'
            ),
            (
                '(((a:1 AND NOT b:2)) OR (NOT c:3))',
                SearchFilter(
                    root=Or(
                        left=And(
                            left=EqualityTerm('a', '1', 'a:1'),
                            right=Not(EqualityTerm('b', '2', 'b:2'))
                        ),
                        right=Not(EqualityTerm('c', '3', 'c:3'))
                    )
                ),
                'extra parantheses'
            ),
            (
                'a:1 AND (b:2 OR c:3)',
                SearchFilter(
                    root=And(
                        left=EqualityTerm('a', '1', 'a:1'),
                        right=Or(
                            left=EqualityTerm('b', '2', 'b:2'),
                            right=EqualityTerm('c', '3', 'c:3')
                        )
                    )
                ),
                'or expression'
            ),
            (
                'a:1 AND NOT (b:2 OR c:3)',
                SearchFilter(
                    root=And(
                        left=EqualityTerm('a', '1', 'a:1'),
                        right=Not(
                            Or(
                                left=EqualityTerm('b', '2', 'b:2'),
                                right=EqualityTerm('c', '3', 'c:3')
                            )
                        )
                    )
                ),
                'not expression'
            ),
            (
                'a:1 AND (b:2 OR (c:3 AND (d:4 OR e:5))) OR d:6',
                SearchFilter(
                    root=Or(
                        left=And(
                            left=EqualityTerm('a', '1', 'a:1'),
                            right=Or(
                                left=EqualityTerm('b', '2', 'b:2'),
                                right=And(
                                    left=EqualityTerm('c', '3', 'c:3'),
                                    right=Or(
                                        left=EqualityTerm('d', '4', 'd:4'),
                                        right=EqualityTerm('e', '5', 'e:5')
                                    )
                                )
                            )
                        ),
                        right=EqualityTerm('d', '6', 'd:6')
                    )
                ),
                'nested filter string'
            ),
            (
                "My\\ Field\\-:(How's it going?)",
                SearchFilter(
                    EqualityTerm('My Field-', "How's it going?", "My\\ Field\\-:(How's it going?)")
                ),
                'escaped field name and value'
            ),
            # Grouped text test
            (
                'a:(hello world)',
                SearchFilter(
                    EqualityTerm('a', 'hello world', 'a:(hello world)')
                ),
                'grouped text'
            ),
            # After colon, parenthesis does NOT define an IN list.
            (
                'a:(1, 2, 3)',
                SearchFilter(
                    EqualityTerm('a', '1, 2, 3', 'a:(1, 2, 3)')
                ),
                'After colon, parenthesis does NOT define an IN list'
            ),
            # IN term divider tests
            (
                'a IN (1, 2, 3)',
                SearchFilter(
                    InTerm('a', ['1', '2', '3'], 'a IN (1,2,3)')
                ),
                'basic int IN term'
            ),
            (
                'a in (1, 2, 3)',
                SearchFilter(
                    InTerm('a', ['1', '2', '3'], 'a IN (1,2,3)')
                ),
                'lowercase in term'
            ),
            (
                'a in (1,    2,         3)',
                SearchFilter(
                    InTerm('a', ['1', '2', '3'], 'a IN (1,2,3)')
                ),
                'in term, whitespaces ignored'
            ),
            (
                'a IN (hello,goodbye)',
                SearchFilter(
                    InTerm('a', ['hello', 'goodbye'], 'a IN (hello,goodbye)')
                ),
                'basic string IN term'
            ),
            (
                'a IN (1, 2, 3) AND b:2',
                SearchFilter(
                    root=And(
                        left=InTerm('a', ['1', '2', '3'], 'a IN (1,2,3)'),
                        right=EqualityTerm('b', '2', 'b:2')
                    )
                ),
                'AND with IN term'
            ),
            (
                'a IN (1, 2, 3) OR b:2',
                SearchFilter(
                    root=Or(
                        left=InTerm('a', ['1', '2', '3'], 'a IN (1,2,3)'),
                        right=EqualityTerm('b', '2', 'b:2')
                    )
                ),
                'OR with IN term'
            ),
            (
                'NOT a IN (1, 2, 3)',
                SearchFilter(
                    root=Not(
                        InTerm('a', ['1', '2', '3'], 'a IN (1,2,3)')
                    ),
                ),
                'NOT IN term'
            ),
            (
                'a IN ((value 1), 2, (value 3))',
                SearchFilter(
                    InTerm('a', ['value 1', '2', 'value 3'], 'a IN ((value 1),2,(value 3))')
                ),
                'grouped IN term'
            ),
            (
                'a IN (hi)',
                SearchFilter(
                    InTerm('a', ['hi'], 'a IN (hi)')
                ),
                'single value IN term'
            ),
            (
                'a IN (1, (value 2), 3) AND NOT b IN (4, (value 5), (value 6))',
                SearchFilter(
                    root=And(
                        left=InTerm('a', ['1', 'value 2', '3'], 'a IN (1,(value 2),3)'),
                        right=Not(
                            InTerm('b', ['4', 'value 5', 'value 6'], 'b IN (4,(value 5),(value 6))')
                        )
                    )
                ),
                'Complex IN term'
            ),
            (
                'a:(value 1) OR (b IN ((2)) AND c:3)',
                SearchFilter(
                    root=Or(
                        left=EqualityTerm('a', 'value 1', 'a:(value 1)'),
                        right=And(
                            left=InTerm('b', ['2'], 'b IN ((2))'),
                            right=EqualityTerm('c', '3', 'c:3')
                        )
                    )
                ),
                'Mixed IN term'
            ),
            (
                'a IN (1, 2,, 3)',
                SearchFilter(
                    InTerm('a', ['1', '2', '', '3'], 'a IN (1,2,,3)')
                ),
                'empty string in IN term list'
            ),
            (
                "text_field_2 IN ((some text), (something else))",
                SearchFilter(
                    InTerm('text_field_2', ['some text', 'something else'],
                           "text_field_2 IN ((some text),(something else))")
                ),
                'spaced IN term'

            ),
            (
                "(float_field_1:[0 TO 1]) AND (text_field_1 in ((some text)))",
                SearchFilter(
                    root=And(
                        left=RangeTerm('float_field_1', 0, 1, 'float_field_1:[0 TO 1]'),
                        right=InTerm('text_field_1', ['some text'], 'text_field_1 IN ((some text))')
                    )),
                "Parenthesis around equality and IN terms"
            ),
            (
                "((float_field_1:[0 TO 1])) AND ((text_field_1 in ((some text))))",
                SearchFilter(
                    root=And(
                        left=RangeTerm('float_field_1', 0, 1, 'float_field_1:[0 TO 1]'),
                        right=InTerm('text_field_1', ['some text'], 'text_field_1 IN ((some text))')
                    )),
                "Even more parenthesis around equality and IN terms"
            ),
            (
                "((float_field_1:[0 TO 1])) AND ((text_field_1 in ((some text)) OR text_field_2 IN (1,2,3)))",
                SearchFilter(
                    root=And(
                        left=RangeTerm('float_field_1', 0, 1, 'float_field_1:[0 TO 1]'),
                        right=Or(
                            left=InTerm('text_field_1', ['some text'], 'text_field_1 IN ((some text))'),
                            right=InTerm('text_field_2', ['1', '2', '3'], 'text_field_2 IN (1,2,3)')
                        )
                    )),
                "Many parenthesis, nested groupings"
            ),
        ]

        for filter_string, expected_filter, msg in test_cases:
            with self.subTest(msg):
                parser = MarqoFilterStringParser()
                parsed_filter = parser.parse(filter_string)
                self.assertEqual(expected_filter, parsed_filter, msg)

    def test_parse_malformedString_fails(self):
        # modifier valid at start or after an operator
        # modifier not valid after term, after modifier, at the end of exp, end of string
        # after modifier, can only have a term
        test_cases = [
            ('AND a:1 OR b:2', 'string starts with operator'),
            ('a:(b))', 'extra parenthesis after grouped text'),
            ('a:1 AND b:2 (OR c:3)', 'expression starts with operator'),
            ('a:1 AND b:2 OR', 'string ends with operator'),
            ('a:1 AND (b:2 OR c:3 AND) OR e:5', 'expression ends with operator'),
            ('a:1 AND b:2 OR OR c:3', 'operator after operator'),
            ('a:1 AND b:2 OR NOT OR c:3', 'operator after modifier'),
            ('a:1 AND b:2 OR NOT', 'string ends with a modifier'),
            ('a:1 AND (b:2 OR c:3 NOT) OR e:5', 'expression ends with a modifier'),
            ('a:1 AND b:2 OR NOT NOT c:3', 'modifier after modifier'),
            ('a:1 NOT a:1', 'modifier after term'),
            ('a:1 a:1', 'term after term'),
            ('(a:1 AND b:2) b:2', 'term after expression'),
            ('(a:1 AND b:2)(c:3 AND d:4)', 'expression after expression'),
            ('a:1 (c:3 AND d:4)', 'expression after term'),
            ('a:1 AND b:2)', 'expression not opened'),
            ('(a:1 AND b:2', 'expression not closed'),
            ('', 'empty expression'),
            (' ', 'empty expression'),
            ('   ', 'empty expression'),
            ('(', '('),
            (')', ')'),
            ('()', '()'),
            ('a:1 AND (b:2 OR (c:3 AND (d:4 OR e:5)) OR d:6', 'imbalanced parentheses, not closed'),
            ('a:1 AND b:2 OR (c:3 AND (d:4 OR e:5))) OR d:6', 'imbalanced parentheses, not opened'),

            # In term tests
            ('a IN (1, 2 OR 3)', 'OR in IN term'),
            ('a IN (1, 2 AND 3)', 'AND in IN term'),
            ('a IN (1, 2 NOT 3)', 'NOT in IN term'),
            ('a IN (1, 2, 3))', 'extra parenthesis in IN term'),
            ('a IN (val1, val 2, val3)', 'ungrouped space in IN term'),
            ('a IN 1, 2, 3)', 'IN term with no opening parenthesis'),
        ]

        for filter_string, msg in test_cases:
            with self.subTest(msg):
                with self.assertRaises(FilterStringParsingError,
                                       msg=f"Did not raise error on malformed filter string '{filter_string}'"):
                    parser = MarqoFilterStringParser()
                    parser.parse(filter_string)

    def test_equality(self):
        """
        Test all equality functions
        """
        tree1 = Or(
            left=And(
                left=EqualityTerm('a', '1', 'a:1'),
                right=EqualityTerm('b', '2', 'b:2')
            ),
            right=EqualityTerm('c', '3', 'c:3')
        )
        # Set one left node to None
        tree2 = Or(
            left=And(
                left=EqualityTerm('b', '2', 'b:2'),
                right=None
            ),
            right=EqualityTerm('c', '3', 'c:3')
        )
        # Change value of one term from tree1
        tree3 = Or(
            left=And(
                left=EqualityTerm('a', '1', 'a:1'),
                right=EqualityTerm('b', '2', 'b:2')
            ),
            right=EqualityTerm('e', '5', 'e:5')
        )
        # Only change operator types from tree1
        tree4 = And(
            left=Or(
                left=EqualityTerm('a', '1', 'a:1'),
                right=EqualityTerm('b', '2', 'b:2')
            ),
            right=EqualityTerm('c', '3', 'c:3')
        )

        trees = [tree1, tree2, tree3, tree4]
        for i in range(len(trees)):
            self.assertTrue(trees[i] == trees[i], f'Self-equality failed for tree {i}')

        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                self.assertFalse(trees[i] == trees[j], f'Equality returned True for unequal trees {i} and {j}')
