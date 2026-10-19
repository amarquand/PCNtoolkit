import json

from pcntoolkit.dataio.fileio import to_json_with_one_line_lists


def test_to_json_with_one_line_lists() -> None:
    """Dicts are indented, each list is on one line, and the values are kept."""
    obj = {
        "name": "roi",
        "settings": {"degree": 3, "tol": 1e-8, "warp": None},
        "m": [0.1, -2.5, 3.0],
        "A": [[1.0, 0.5], [0.5, 2.0]],
        "empty": {},
    }
    text = to_json_with_one_line_lists(obj)
    assert json.loads(text) == obj
    lines = text.splitlines()
    assert '        "degree": 3,' in lines
    assert '    "m": [0.1, -2.5, 3.0],' in lines
    assert '    "A": [[1.0, 0.5], [0.5, 2.0]],' in lines


def test_to_json_with_one_line_lists_null_character() -> None:
    """A string that looks like a placeholder does not corrupt the output."""
    obj = {"name": "\x000\x00", "m": [1.0, 2.0]}
    assert json.loads(to_json_with_one_line_lists(obj)) == obj
