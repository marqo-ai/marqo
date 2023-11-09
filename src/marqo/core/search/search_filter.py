from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, List

from marqo.core.exceptions import FilterStringParsingError
from marqo.exceptions import InternalError


class Node(ABC):
    def __init__(self, raw: str):
        self.raw = raw


class Term(Node, ABC):
    def __init__(self, field: str, raw: str):
        super().__init__(raw)
        self.field = field

    def __str__(self):
        return self.raw


class Operator(Node, ABC):
    def __init__(self, left: Node, right: Node, raw: str):
        super().__init__(raw)
        self.left = left
        self.right = right

    @property
    @abstractmethod
    def greedy(self) -> bool:
        pass

    def __eq__(self, other):
        return (
                type(self) == type(other) and
                self.left == other.left and
                self.right == other.right and
                self.raw == other.raw
        )

    def __str__(self):
        if self.greedy:
            return f'{str(self.left)} {self.raw} {str(self.right)}'
        else:
            return f'({str(self.left)} {self.raw} {str(self.right)})'


class Modifier(Node, ABC):
    def __init__(self, modified: Union[Term, Operator], raw: str):
        super().__init__(raw)
        self.modified = modified

        if isinstance(modified, Operator) and modified.right is None:
            raise ValueError(f'A modifier can only be applied to terms and expressions '
                             f'(operator with both left and right)')

    def __eq__(self, other):
        return type(self) == type(other) and self.modified == other.modified and self.raw == other.raw

    def __str__(self):
        return f'{self.raw} ({str(self.modified)})'


class EqualityTerm(Term):
    def __init__(self, field: str, value: str, raw: str):
        super().__init__(field, raw)
        self.value = value

    def __eq__(self, other):
        return (
                type(self) == type(other) and
                self.field == other.field and
                self.value == other.value and
                self.raw == other.raw
        )


RangeLimit = Union[int, float]


class RangeTerm(Term):
    def __init__(self, field: str, lower: Optional[RangeLimit], upper: Optional[RangeLimit], raw: str):
        super().__init__(field, raw)
        self.lower = lower
        self.upper = upper

        if lower is None and upper is None:
            raise ValueError(f'At least one of lower or upper must be specified')

    def __eq__(self, other):
        return (
                type(self) == type(other) and
                self.field == other.field and
                self.lower == other.lower and
                self.upper == other.upper and
                self.raw == other.raw
        )

    @classmethod
    def parse(cls, field: str, value: str, raw: str) -> "RangeTerm":
        # Value must be in the form of 'lower TO upper' (brackets have been stripped)
        lower_str, upper_str = value.lower().split(' to ')

        def parse_limit(limit: str):
            try:
                return int(limit)
            except ValueError:
                try:
                    return float(limit)
                except ValueError:
                    raise ValueError(f"Invalid range limit '{limit}'")

        lower = None
        if lower_str != '*':
            lower = parse_limit(lower_str)
        upper = None
        if upper_str != '*':
            upper = parse_limit(upper_str)

        return cls(field, lower, upper, raw)


class And(Operator):
    def __init__(self, left: Node, right: Node):
        super().__init__(left, right, 'AND')

    greedy = True


class Or(Operator):
    def __init__(self, left: Node, right: Node):
        super().__init__(left, right, 'OR')

    greedy = False


class Not(Modifier):
    def __init__(self, modified: Union[Term, Operator]):
        super().__init__(modified, 'NOT')


class SearchFilter:
    """
    A search filter as a tree of Nodes.

    To generate a filter string from a SearchFilter object, traverse the tree in-order for Operators and pre-order
    for Modifiers. Terms are always leaf nodes.
    """

    def __init__(self, root: Node):
        self.root = root

    def __eq__(self, other):
        return type(self) == type(other) and self.root == other.root

    def __str__(self) -> str:
        if not self.root:
            return ''

        filter_string = str(self.root)
        if filter_string.startswith('(') and filter_string.endswith(')'):
            return filter_string[1:-1]
        return filter_string


class MarqoFilterStringParser:
    """
    This class parses a Marqo filter string into a SearchFilter object.

    This class is not thread-safe.
    """

    # Terminology:
    #  Term: A single term in the form of 'field:value' or 'field:[lower|* TO upper|*]'
    #  Expression:
    #       * A term or,
    #       * A combination of terms and operators e.g., 'a:1 AND b:2' or,
    #       * A modifier and an expression e.g., 'NOT (a:1 AND b:2)', 'NOT a:1'
    #  Operator: Instance of Operator class (And, Or) with a left but no right (otherwise it's an expression)
    #  Modifier: Instance of Modifier class (Not) without a modified (otherwise it's an expression)
    #  Token: A single character or a sequence of characters read from the filter string

    class _TermType(Enum):
        Equality = 1
        Range = 2

    def parse(self, filter_string: str) -> SearchFilter:
        self._reset_state()

        if filter_string == '':
            raise FilterStringParsingError('Cannot parse empty filter string')

        stack: List[Union[str, Operator, Term]] = []
        parenthesis_count = 0
        escape = False
        read_space_until = None  # ignore space until reaching the value of this variable

        for i in range(len(filter_string)):
            c = filter_string[i]

            # Special processing if we are reading a term value
            if self._read_term_value and not (c in [' ', ')'] and not escape and read_space_until is None):
                if self._reached_term_end:
                    self._error(f"Expected end of term, but found '{c}'", filter_string, i)

                if escape:
                    self._current_token.append(c)
                    self._term_value.append(c)
                    escape = False
                elif c == '\\':
                    escape = True
                else:
                    if c == read_space_until:
                        read_space_until = None
                        self._reached_term_end = True
                    elif len(self._term_value) == 0 and c == '(' and not read_space_until:  # start of term value
                        read_space_until = ')'
                    elif len(self._term_value) == 0 and c == '[' and not read_space_until:  # start of term value
                        read_space_until = ']'
                        self._term_type = MarqoFilterStringParser._TermType.Range
                    else:
                        self._term_value.append(c)

                    self._current_token.append(c)

                continue

            if escape:
                self._current_token.append(c)
                escape = False
            elif c == ':':
                self._read_term_value = True
                self._term_type = MarqoFilterStringParser._TermType.Equality
                self._term_field = ''.join(self._current_token)
                self._current_token.append(c)
            elif c == '(':
                if len(self._current_token) > 0:
                    self._error('Unexpected (', filter_string, i)

                stack.append(c)
                parenthesis_count += 1
            elif c == ')':
                self._push_token(
                    stack,
                    filter_string,
                    self._current_token,
                    self._term_field,
                    self._term_value,
                    self._term_type,
                    i
                )
                self._reset_state()

                self._merge_expression(stack, filter_string, i)
                parenthesis_count -= 1
            elif c == '\\':
                escape = True
            elif c == ' ':
                self._push_token(
                    stack,
                    filter_string,
                    self._current_token,
                    self._term_field,
                    self._term_value,
                    self._term_type,
                    i
                )
                self._reset_state()
            else:
                self._current_token.append(c)

        if len(self._current_token) > 0:
            self._push_token(
                stack,
                filter_string,
                self._current_token,
                self._term_field,
                self._term_value,
                self._term_type,
                len(filter_string)
            )

        if parenthesis_count != 0:
            # merge_stack will catch this, but this is a more specific error message
            self._error('Unbalanced parentheses')

        self._merge_stack(stack, filter_string)

        if len(stack) != 1:
            # This should be unreachable
            self._error('Failed to parse filter string')

        root = stack.pop()

        return SearchFilter(root)

    def _push_token(self,
                    stack: List[Union[str, Node]],
                    filter_string: str,
                    current_token: List[str],
                    term_field: Optional[str],
                    term_value: Optional[List[str]],
                    term_type: Optional[_TermType],
                    pos: int):
        if len(current_token) == 0:
            return

        token = ''.join(current_token)

        prev = stack[-1] if len(stack) > 0 else None
        if token == 'AND':
            # Operator must come after an expression
            if not self._is_expression(prev):
                if not isinstance(prev, Node):
                    # Operator at the beginning of expression
                    self._error('Unexpected AND', filter_string, pos - 3)
                else:
                    # Consecutive operators or operator after modifier
                    self._error(f'Expected term or expression, but found AND', filter_string, pos - 3)

            stack.append(And(stack.pop(), None))
        elif token == 'OR':
            # Operator must come after an expression
            if not self._is_expression(prev):
                if not isinstance(prev, Node):
                    # Operator at the beginning of expression
                    self._error('Unexpected OR', filter_string, pos - 2)
                else:
                    # Consecutive operators or operator after modifier
                    self._error(f'Expected term or expression, but found OR', filter_string, pos - 2)

            stack.append(Or(stack.pop(), None))
        elif token == 'NOT':
            # Modifier must come at the beginning of an expression or after an operator
            if not (self._is_start_of_expression(prev) or self._is_operator(prev)):
                # Modifier after modifier or expression
                self._error(
                    f"Unexpected modifier '{token}'",
                    filter_string, pos - len(token)
                )

            stack.append(Not(None))
        else:
            # Term
            if not term_field or not term_value:
                self._error(f"Cannot parse token '{token}'", filter_string, pos)

            term_value = ''.join(term_value)

            # Term must come at the beginning of an expression, after a modifier or after an operator
            if not (self._is_start_of_expression(prev) or self._is_modifier(prev) or self._is_operator(prev)):
                # Term after expression
                self._error(f"Unexpected term '{token}'. Expected an operator", filter_string, pos - len(token))

            node = None
            if term_type == self._TermType.Equality:
                node = EqualityTerm(term_field, term_value, token)
            elif term_type == self._TermType.Range:
                try:
                    node = RangeTerm.parse(term_field, term_value, token)
                except ValueError as e:
                    self._error(f"Cannot parse range term '{token}': {str(e)}", filter_string, pos)
            else:
                raise InternalError(f'Unexpected term type {term_type}')

            if isinstance(prev, Modifier):
                prev.modified = node
                return
            elif isinstance(prev, Operator):
                operator = prev
                if operator.greedy:
                    # AND is our only greedy operator
                    operator.right = node
                    return

            # Not coming after AND or NOT, so just push into the stack
            stack.append(node)

    def _merge_expression(self, stack: List[Union[str, Node]], filter_string: str, pos: int):
        # Expression must end with a term or an expression
        last = stack[-1] if len(stack) > 0 else None
        if not self._is_expression(last):
            if last is None:
                self._error('Unexpected )', filter_string, pos)
            elif last == '(':
                # Empty expression
                self._error('Empty expression', filter_string, pos - 1)
            else:
                # Expression ending with operator
                self._error(
                    f'Expected term or expression, but found {last.raw}',
                    filter_string, pos - len(last.raw)
                )

        while len(stack) > 1:
            node = stack.pop()

            if not isinstance(node, Node):
                # Expected to be unreachable
                self._error(
                    f"Unexpected token '{node}' in expression ending at position {pos}",
                    filter_string, pos
                )

            prev = stack[-1]
            if prev == '(':
                # We've reached the start of the parenthetical expression
                stack.pop()

                # If there's a modifier for this expression, apply it
                if len(stack) > 0 and isinstance(stack[-1], Modifier):
                    modifier = stack.pop()
                    modifier.modified = node
                    node = modifier

                # Expression must come at the beginning of an expression or after an operator
                if not (len(stack) == 0 or stack[-1] == '(' or self._is_operator(stack[-1])):
                    self._error(
                        f'Unexpected expression ending at position {pos}',
                        filter_string, pos
                    )

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
                self._error(f'Unexpected term {prev.raw} in expression ending at position {pos}')

            prev.right = node

        # No corresponding ( if we get here
        self._error('Unexpected )', filter_string, pos)

    def _merge_stack(self, stack: List[Union[str, Node]], filter_string: str):
        if len(stack) == 0:
            # Expected to be unreachable due to earlier checks
            self._error('Empty filter string')

        # The string (expression) must end with an expression
        last = stack[-1]
        if not self._is_expression(last):
            if not isinstance(last, Node):
                # Expected to be unreachable
                self._error(f"Unexpected token '{last}'")
            self._error(
                f'Expected term or expression, but found {str(last.raw)}',
                filter_string, len(filter_string) - len(str(last.raw))
            )

        while len(stack) > 1:
            node = stack.pop()

            if not isinstance(node, Node):
                # Expected to be unreachable
                self._error(f"Unexpected token '{node}'")

            prev = stack[-1]
            if not isinstance(prev, Operator):
                # Expected to be unreachable due to parenthesis balance check prior to this
                self._error(f"Unexpected token '{prev}'")

            prev.right = node

    def _is_expression(self, node: Node):
        return (
                isinstance(node, Term) or
                isinstance(node, Operator) and node.right is not None or
                isinstance(node, Modifier) and node.modified is not None
        )

    def _is_start_of_expression(self, node: Node):
        return node is None or node == '('

    def _is_operator(self, node: Node):
        return isinstance(node, Operator) and node.right is None

    def _is_modifier(self, node: Node):
        return isinstance(node, Modifier) and node.modified is None

    def _error(self, msg, filter_string: Optional[str] = None, pos: Optional[int] = None):
        if pos is not None and filter_string is not None:
            prefix = '\nError parsing filter: '
            error_msg = f'{prefix}{filter_string}\n'
            error_msg += ' ' * (pos + len("Error parsing filter: ")) + '^\n'
            error_msg += f'at position {pos}: {msg}'

            raise FilterStringParsingError(error_msg)
        else:
            raise FilterStringParsingError(f'Error parsing filter string {pos}: {msg}')

    def _reset_state(self):
        self._current_token: List[str] = []
        self._term_field: Optional[str] = None
        self._term_value: List[str] = []
        self._read_term_value: bool = False
        self._reached_term_end: bool = False
        self._term_type: Optional[MarqoFilterStringParser._TermType] = None
