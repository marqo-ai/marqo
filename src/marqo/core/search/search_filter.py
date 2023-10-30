from abc import ABC, abstractmethod
from typing import Optional, Union, List

from marqo.core.exceptions import FilterStringParsingError


class Node(ABC):
    pass


class Term(Node):
    def __init__(self, term: str):
        self.term = term

    def __str__(self):
        return self.term

    def __eq__(self, other):
        return type(self) == type(other) and self.term == other.term


class Operator(Node, ABC):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    @abstractmethod
    def greedy(self) -> bool:
        pass

    def __eq__(self, other):
        return type(self) == type(other) and self.left == other.left and self.right == other.right


class And(Operator):
    greedy = True

    def __str__(self):
        return 'AND'


class Or(Operator):
    greedy = False

    def __str__(self):
        return 'OR'


class SearchFilter:
    """
    A search filter as a tree of nodes that can be traversed in-order to translate the filter to another DSL
    such as Vespa YQL.
    """

    def __init__(self, root: Node):
        self.root = root

    def __eq__(self, other):
        return type(self) == type(other) and self.root == other.root

    def print_tree(self) -> str:
        if not self.root:
            return ''

        def _print_node(node):
            if isinstance(node, Term):
                return str(node)
            elif isinstance(node, Operator):
                if node.greedy:
                    return f'{_print_node(node.left)} {str(node)} {_print_node(node.right)}'
                else:
                    return f'({_print_node(node.left)} {str(node)} {_print_node(node.right)})'
            else:
                raise Exception(f'Unexpected node type {type(node)}')

        printed_filter = _print_node(self.root)
        if printed_filter.startswith('(') and printed_filter.endswith(')'):
            return printed_filter[1:-1]
        return printed_filter


class MarqoFilterStringParser:
    """
    This class parses a Marqo filter string into a SearchFilter object.
    """

    @classmethod
    def parse(cls, filter_string: str) -> SearchFilter:
        if filter_string == '':
            raise FilterStringParsingError('Cannot parse empty filter string')

        stack: List[Union[str, Operator, Term]] = []

        def push_token(token, pos):
            if token == '':
                return

            prev = stack[-1] if len(stack) > 0 else None
            if token == 'AND':
                # Operator must come after a term or an expression
                if not (is_term(prev) or is_expression(prev)):
                    if not isinstance(prev, Node):
                        # Operator at the beginning of expression
                        error('Unexpected AND', pos - 3)
                    else:
                        # Consecutive operators
                        error(f'Expected term or expression, but found AND', pos - 3)

                stack.append(And(stack.pop(), None))
            elif token == 'OR':
                # Operator must come after a term or an expression
                if not (is_term(prev) or is_expression(prev)):
                    if not isinstance(prev, Node):
                        # Operator at the beginning of expression
                        error('Unexpected OR', pos - 2)
                    else:
                        # Consecutive operators
                        error(f'Expected term or expression, but found OR', pos - 2)

                stack.append(Or(stack.pop(), None))
            else:
                # Term must come at the beginning of an expression or after an operator
                if not (prev is None or prev == '(' or is_operator(prev)):
                    # Term after term or expression
                    error(f"Unexpected term '{token}. Expected an operator", pos - len(token))

                node = Term(token)
                if isinstance(prev, Operator):
                    operator = prev
                    if operator.greedy:
                        # AND is our only greedy operator
                        operator.right = node
                        return

                # Not coming after AND, so just push into the stack
                stack.append(Term(token))

        def merge_expression(pos):
            # Expression must end with a term or an expression
            last = stack[-1] if len(stack) > 0 else None
            if not (is_term(last) or is_expression(last)):
                if last is None:
                    error('Unexpected )', pos)
                elif last == '(':
                    # Empty expression
                    error('Empty expression', pos - 1)
                else:
                    # Expression ending with operator
                    error(f'Expected term or expression, but found {str(last)}', pos - len(str(last)))

            while len(stack) > 1:
                node = stack.pop()

                if not isinstance(node, Node):
                    # Expected to be unreachable
                    error(f"Unexpected token '{node}' in expression ending at position {pos}", pos)

                prev = stack[-1]
                if prev == '(':
                    # We've reached the start of the parenthetical expression
                    stack.pop()

                    # Expression must come at the beginning of an expression or after an operator
                    if not (len(stack) == 0 or stack[-1] == '(' or is_operator(stack[-1])):
                        error(f'Unexpected expression ending at position {pos}', pos)

                    # Check if the previous operator is greedy (AND)
                    if len(stack) > 0 and isinstance(stack[-1], Operator):
                        operator = stack[-1]

                        if operator.greedy:
                            operator.right = node
                            return

                    stack.append(node)
                    return

                if not isinstance(prev, Operator):
                    # Expected to be unreachable
                    error(f'Unexpected term {prev.term} in expression ending at position {pos}')

                prev.right = node

            # No corresponding ( if we get here
            error('Unexpected )', pos)

        def merge_stack():
            if len(stack) == 0:
                # Expected to be unreachable due to earlier checks
                error('Empty filter string')

            # The string (expression) must end with a term or an expression
            last = stack[-1]
            if not (is_term(last) or is_expression(last)):
                error(f'Expected term or expression, but found {str(last)}', len(filter_string) - len(str(last)))

            while len(stack) > 1:
                node = stack.pop()

                if not isinstance(node, Node):
                    # Expected to be unreachable
                    error(f"Unexpected token '{node}'")

                prev = stack[-1]
                if not isinstance(prev, Operator):
                    # Expected to be unreachable due to parenthesis balance check prior to this
                    error(f"Unexpected token '{prev}'")

                prev.right = node

        def is_expression(node: Node):
            return isinstance(node, Operator) and node.right is not None

        def is_operator(node: Node):
            return isinstance(node, Operator) and node.right is None

        def is_term(node: Node):
            return isinstance(node, Term)

        def error(msg, pos: Optional[int] = None):
            if pos is not None:
                prefix = '\nError parsing filter: '
                error_msg = f'{prefix}{filter_string}\n'
                error_msg += ' ' * (pos + len("Error parsing filter: ")) + '^\n'
                error_msg += f'at position {pos}: {msg}'

                raise FilterStringParsingError(error_msg)
            else:
                raise FilterStringParsingError(f'Error parsing filter string {pos}: {msg}')

        current_token = ''
        parenthesis_count = 0
        escape = False
        for i in range(len(filter_string)):
            c = filter_string[i]
            if escape:
                current_token += c
                escape = False
            elif c == '(':
                if current_token != '':
                    error('Unexpected (', i)

                stack.append(c)
                parenthesis_count += 1
            elif c == ')':
                push_token(current_token, i)
                current_token = ''

                merge_expression(i)
                parenthesis_count -= 1
            elif c == '\\':
                escape = True
            elif c == ' ':
                push_token(current_token, i)
                current_token = ''
            else:
                current_token += c

        if current_token != '':
            push_token(current_token, len(filter_string))

        if parenthesis_count != 0:
            # merge_stack will catch this, but this is a more specific error message
            error('Unbalanced parentheses')

        merge_stack()
        if len(stack) != 1:
            # This should be unreachable
            error('Failed to parse filter string')

        root = stack.pop()

        return SearchFilter(root)
