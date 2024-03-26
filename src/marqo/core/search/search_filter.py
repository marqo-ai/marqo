from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, List

from marqo.core.exceptions import FilterStringParsingError
from marqo.exceptions import InternalError


class Node(ABC):
    def __init__(self, raw: str):
        """
        Initialize a Node object.

        Args:
            raw: The raw user input string that this node was parsed from. This is used for error messages and
            debugging only and must not be used when interpreting the semantics of the node.
        """
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

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.left)}, {repr(self.right)}, {repr(self.raw)})'


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

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.modified)}, {repr(self.raw)})'


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

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.field)}, {repr(self.value)}, {repr(self.raw)})'


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

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.field)}, {repr(self.lower)}, {repr(self.upper)}, ' \
               f'{repr(self.raw)})'

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


class InTerm(Term):
    def __init__(self, field: str, value_list: List[str], raw: str):
        super().__init__(field, raw)
        self.value_list = value_list

    def __eq__(self, other):
        return (
                type(self) == type(other) and
                self.field == other.field and
                # Order is disregarded for IN lists.
                set(self.value_list) == set(other.value_list) and
                self.raw == other.raw
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.field)}, {repr(self.value_list)}, {repr(self.raw)})'


class And(Operator):
    def __init__(self, left: Node, right: Node, raw: str = 'AND'):
        super().__init__(left, right, raw)

    greedy = True


class Or(Operator):
    def __init__(self, left: Node, right: Node, raw: str = 'OR'):
        super().__init__(left, right, raw)

    greedy = False


class Not(Modifier):
    def __init__(self, modified: Union[Term, Operator], raw: str = 'NOT'):
        super().__init__(modified, raw)


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

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.root)})'


class MarqoFilterStringParser:
    """
    This class parses a Marqo filter string into a SearchFilter object.

    This class is not thread-safe.
    """

    # Terminology:
    #  Term: A single term in the form of:
    #       a) 'field:value'
    #       b) 'field:[lower|* TO upper|*]'
    #       c) 'field IN (value1, value2, ...)'
    #  Term Divider: Character/s that separate term into field and value. Either ':' or ' IN '.
    #  Expression:
    #       * A term or,
    #       * A combination of terms and operators e.g., 'a:1 AND b:2' or,
    #       * A modifier and an expression e.g., 'NOT (a:1 AND b:2)', 'NOT a:1'
    #  Operator: Instance of Operator class (And, Or) with a left but no right (otherwise it's an expression)
    #  Modifier: Instance of Modifier class (Not) without a modified (otherwise it's an expression)
    #  Token: A single character or a sequence of characters read from the filter string

    IN_TERM_DIVIDER = ' IN ('

    class _TermType(Enum):
        Equality = 1
        Range = 2
        In = 3

    def _term_divider_is_IN(self, i: int, filter_string: str) -> bool:
        """
        Given 'i' and the full filter string, determine if 'i' is at the beginning of the IN term divider.
        If any of the spaces or parenthesis are missing, this will return False.
        """
        candidate_substring = filter_string[i:i + len(self.IN_TERM_DIVIDER)].upper()
        # Extra check if substring has " IN " but is missing the open parenthesis
        if candidate_substring[:4] == ' IN ' and candidate_substring[4] != '(':
            self._error('Expected open parenthesis after " IN "', filter_string, i)

        return candidate_substring == self.IN_TERM_DIVIDER

    def _append_to_term_value(self, c: str):
        """
        If the term type is IN, append c to the last list in term_value.
        Otherwise, append c to term_value.

        For IN terms, term_value is a list of lists.
        """
        if self._term_type == MarqoFilterStringParser._TermType.In:
            self._term_value[-1].append(c)
        else:
            self._term_value.append(c)

    def _get_current_term_value(self):
        """
        If the term type is IN, return the last list in term_value.
        Otherwise, return term_value.

        For IN terms, term_value is a list of lists.
        """
        if self._term_type == MarqoFilterStringParser._TermType.In:
            return self._term_value[-1]
        else:
            return self._term_value

    def _join_term_value(self, term_value: Union[List[str], List[List[str]]]) -> Union[str, List[str]]:
        """
        If the term type is IN, join every list in term_value
        Otherwise, join term_value.

        For IN terms, term_value is a list of lists.
        """
        if self._term_type == MarqoFilterStringParser._TermType.In:
            return [''.join(item) for item in term_value]
        else:
            return ''.join(term_value)

    def parse(self, filter_string: str) -> SearchFilter:
        self._reset_state()

        if filter_string == '':
            raise FilterStringParsingError('Cannot parse empty filter string')

        stack: List[Union[str, Operator, Term]] = []
        parenthesis_count = 0
        escape = False
        read_space_until = None  # ignore space until reaching the value of this variable

        i = 0
        while i < len(filter_string):
            c = filter_string[i]

            # Stop ignoring white spaces if we are not in a list after a comma
            if c != ' ':
                self._in_term_value_after_comma = False

            # Special processing if we are reading a term value
            if self._read_term_value and (
                    c not in [' ', ')'] or
                    read_space_until != None or
                    self._in_term_value_after_comma or
                    escape
            ):
                if self._reached_term_end:
                    self._error(f"Expected end of term, but found '{c}'", filter_string, i)

                if escape:
                    self._current_token.append(c)
                    self._current_raw_token.append(c)
                    self._append_to_term_value(c)
                    escape = False
                elif c == '\\':
                    self._current_raw_token.append(c)
                    escape = True
                elif c == ',' and self._term_type == MarqoFilterStringParser._TermType.In:
                    # Comma only has special meaning in IN term lists.
                    # Break into the next list for term_value.
                    self._term_value.append([])
                    self._current_token.append(c)
                    self._current_raw_token.append(c)
                    self._in_term_value_after_comma = True
                elif self._in_term_value_after_comma and c == ' ':
                    # Ignore all whitespace after comma in IN term lists.
                    pass
                else:
                    if c == read_space_until:
                        read_space_until = None
                        # Term end is reached if ) or ] are reached, EXCEPT for grouped items in IN lists.
                        # Example: a IN (1, 2, (3, 4))
                        # In this case, reaching the ) after 4 does NOT end the term.
                        if not (self._term_type == MarqoFilterStringParser._TermType.In and c == ')'):
                            self._reached_term_end = True
                    elif len(self._get_current_term_value()) == 0 and c == '(' and not read_space_until:  # start of term value
                        read_space_until = ')'
                    elif len(self._get_current_term_value()) == 0 and c == '[' and not read_space_until:  # start of term value
                        read_space_until = ']'
                        if self._term_type != MarqoFilterStringParser._TermType.In:
                            self._term_type = MarqoFilterStringParser._TermType.Range
                        else:
                            self._error('Unexpected [ after IN operator.', filter_string, i)
                    else:
                        self._append_to_term_value(c)

                    self._current_token.append(c)
                    self._current_raw_token.append(c)

                # Increment i for loop
                i += 1
                continue

            if escape:
                self._current_token.append(c)
                self._current_raw_token.append(c)
                escape = False
            elif c == ':':
                self._read_term_value = True
                self._term_type = MarqoFilterStringParser._TermType.Equality
                self._term_field = ''.join(self._current_token)
                self._current_token.append(c)
                self._current_raw_token.append(c)
            elif c == '(':
                if len(self._current_token) > 0:
                    self._error('Unexpected (', filter_string, i)

                stack.append(c)
                parenthesis_count += 1
            elif c == ')':
                parenthesis_count -= 1
                # Last parenthesis in an IN list
                # Example: a IN (1, 2, 3)
                # Note: token is NOT pushed at this state. Just note that term has ended.
                if self._term_type == MarqoFilterStringParser._TermType.In and not self._reached_term_end:
                    self._current_token.append(c)
                    self._current_raw_token.append(c)
                    self._reached_term_end = True
                else:
                    # Grouping parenthesis for expressions (grouping multiple terms)
                    # Example: (a:1 AND b in (2)) OR c:3
                    self._push_token(
                        stack,
                        filter_string,
                        self._current_token,
                        self._current_raw_token,
                        self._term_field,
                        self._term_value,
                        self._term_type,
                        i
                    )
                    self._reset_state()
                    self._merge_expression(stack, filter_string, i)

            elif c == '\\':
                self._current_raw_token.append(c)
                escape = True
            elif c == ' ':
                # Found the ' IN ' operator. Look for a list starting with '(' on the next pass.
                if self._term_divider_is_IN(i, filter_string):
                    self._read_term_value = True
                    self._term_type = MarqoFilterStringParser._TermType.In
                    self._term_value = [[]]
                    self._term_field = ''.join(self._current_token)
                    self._current_token.append(self.IN_TERM_DIVIDER)
                    self._current_raw_token.append(self.IN_TERM_DIVIDER)
                    parenthesis_count += 1

                    # Skip the next 4 characters (they were used in ' IN (' operator).
                    i += len(self.IN_TERM_DIVIDER) - 1

                elif not self._reached_term_end and self._term_type == MarqoFilterStringParser._TermType.In:
                    # Found UNGROUPED white space in IN term list. This should be an error.
                    self._error('Unexpected white space in IN term list', filter_string, i)
                else:
                    # Found the space AFTER a full term/token.
                    self._push_token(
                        stack,
                        filter_string,
                        self._current_token,
                        self._current_raw_token,
                        self._term_field,
                        self._term_value,
                        self._term_type,
                        i
                    )
                    self._reset_state()
            else:
                self._current_token.append(c)
                self._current_raw_token.append(c)
            # Increment i for loop
            i += 1

        # Post-parse loop cleanup
        if len(self._current_token) > 0:
            self._push_token(
                stack,
                filter_string,
                self._current_token,
                self._current_raw_token,
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
                    current_raw_token: List[str],
                    term_field: Optional[str],
                    term_value: Optional[Union[List[str], List[List[str]]]],
                    term_type: Optional[_TermType],
                    pos: int):
        if len(current_token) == 0:
            return

        token = ''.join(current_token)
        raw_token = ''.join(current_raw_token)

        prev = stack[-1] if len(stack) > 0 else None
        if token == 'AND':
            # Operator must come after an expression
            if not self._is_expression(prev):
                if not isinstance(prev, Node):
                    # Operator at the beginning of expression
                    self._error('Unexpected AND', filter_string, pos - len(raw_token))
                else:
                    # Consecutive operators or operator after modifier
                    self._error(f'Expected term or expression, but found AND', filter_string, pos - len(raw_token))

            stack.append(And(stack.pop(), None, raw_token))
        elif token == 'OR':
            # Operator must come after an expression
            if not self._is_expression(prev):
                if not isinstance(prev, Node):
                    # Operator at the beginning of expression
                    self._error('Unexpected OR', filter_string, pos - len(raw_token))
                else:
                    # Consecutive operators or operator after modifier
                    self._error(f'Expected term or expression, but found OR', filter_string, pos - len(raw_token))

            stack.append(Or(stack.pop(), None, raw_token))
        elif token == 'NOT':
            # Modifier must come at the beginning of an expression or after an operator
            if not (self._is_start_of_expression(prev) or self._is_operator(prev)):
                # Modifier after modifier or expression
                self._error(
                    f"Unexpected modifier '{token}'",
                    filter_string, pos - len(raw_token)
                )

            stack.append(Not(None, raw_token))
        else:
            # Term
            if not term_field or not term_value:
                self._error(f"Cannot parse token '{token}'", filter_string, pos)

            term_value = self._join_term_value(term_value)

            # Term must come at the beginning of an expression, after a modifier or after an operator
            if not (self._is_start_of_expression(prev) or self._is_modifier(prev) or self._is_operator(prev)):
                # Term after expression
                self._error(
                    f"Unexpected term '{token}'. Expected an operator", filter_string, pos - len(raw_token)
                )

            node = None
            if term_type == self._TermType.Equality:
                node = EqualityTerm(term_field, term_value, raw_token)
            elif term_type == self._TermType.Range:
                try:
                    node = RangeTerm.parse(term_field, term_value, raw_token)
                except ValueError as e:
                    self._error(f"Cannot parse range term '{token}': {str(e)}", filter_string, pos)
            elif term_type == self._TermType.In:
                # For IN terms, term_value should already be a list, not a string
                node = InTerm(term_field, term_value, raw_token)
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
        self._current_raw_token: List[str] = []
        self._term_field: Optional[str] = None
        self._term_value: Union[List[str], List[List[str]]] = []
        self._read_term_value: bool = False
        self._reached_term_end: bool = False
        self._term_type: Optional[MarqoFilterStringParser._TermType] = None
        self._in_term_value_after_comma: bool = False
