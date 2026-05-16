"""Tests for the code claim AST/bracket validator."""
from src.verification.code_validator import check_brackets, check_code, check_python


def test_python_valid():
    r = check_python("def f(x):\n    return x + 1\n")
    assert r.parseable
    assert r.language == "python"


def test_python_invalid():
    r = check_python("def f(x:\n    return x + 1\n")
    assert not r.parseable
    assert "SyntaxError" in r.reason


def test_brackets_balanced():
    r = check_brackets("function foo() { return [1, 2, 3]; }")
    assert r.parseable


def test_brackets_unbalanced():
    r = check_brackets("function foo() { return [1, 2, 3; }")
    assert not r.parseable


def test_check_code_strips_fence():
    r = check_code("```python\nimport torch\n```")
    assert r.parseable
    assert r.language == "python"


def test_check_code_empty():
    r = check_code("")
    assert not r.parseable
    assert r.reason == "empty"


def test_check_code_string_contains_brackets():
    # The bracket checker should not be fooled by brackets inside strings.
    r = check_brackets('let s = "([{ unbalanced text }";')
    assert r.parseable


def test_python_hint_overrides():
    # Even without a fence, hint can force python parsing
    r = check_code("x = [1,2", language_hint="python")
    assert not r.parseable
    assert r.language == "python"
