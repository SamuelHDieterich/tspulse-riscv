"""Evaluate the byte formulas of `memory.csv`, which are sums and products of integers and the channel count."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import ast
import operator
from collections.abc import Callable


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The one name a formula may use
CHANNELS_NAME = "channels"

# The operators a formula may use
OPERATORS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Mult: operator.mul,
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _walk(node: ast.expr, channels: int) -> int:
    """Evaluate one node of a formula's syntax tree.

    Parameters
    ----------
    node : ast.expr
        The node.
    channels : int
        The value of `channels`.

    Returns
    -------
    int
        The node's value.

    Raises
    ------
    ValueError
        When the node is anything but an integer, `channels`, `*` or `+`.
    """

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.Name) and node.id == CHANNELS_NAME:
        return channels
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](
            _walk(node.left, channels), _walk(node.right, channels)
        )
    raise ValueError(f"not allowed in a formula: {ast.dump(node)}")


def evaluate_formula(formula: str, channels: int) -> int:
    """Evaluate a byte formula such as `2 * channels * 8`, without `eval`.

    Parameters
    ----------
    formula : str
        Integers, the name `channels`, `*` and `+`.
    channels : int
        The value of `channels`.

    Returns
    -------
    int
        The formula's value.

    Raises
    ------
    ValueError
        When the formula isn't an expression or uses anything beyond what is allowed.
    """

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"not a formula: {formula!r}") from error
    return _walk(tree.body, channels)
