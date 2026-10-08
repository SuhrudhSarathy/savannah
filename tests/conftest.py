"""Test tiers, assigned by top-level directory under tests/.

    pytest                   unit tests only: dummy backbones, no downloads, no sims
    pytest -m smoke          tests/smoke/: short full training runs via scripts/*.py
    pytest -m integration    real pretrained backbones / Hub datasets (downloads)
    pytest -m sim            rollouts in real simulators (PushT, MetaWorld, ...)
    pytest -m ""             everything

Directories whose marker the active `-m` expression excludes are skipped at
collection time, not just deselected afterwards -- so their (heavy, optional)
imports such as simulators and HF models never run in a plain `pytest`.
"""

from pathlib import Path

from _pytest.mark.expression import Expression

TESTS_DIR = Path(__file__).parent

DIR_MARKERS = {
    "smoke": "smoke",
    "integration": "integration",
    "backbones": "integration",
    "data": "integration",
    "eval": "sim",
}


def _dir_marker(path: Path) -> str | None:
    try:
        top = path.relative_to(TESTS_DIR).parts[0]
    except (ValueError, IndexError):
        return None
    return DIR_MARKERS.get(top)


def pytest_ignore_collect(collection_path, config):
    marker = _dir_marker(collection_path)
    markexpr = config.option.markexpr
    if marker is None or not markexpr:
        return None
    selected = Expression.compile(markexpr).evaluate(
        lambda name, **kwargs: name == marker
    )
    return True if not selected else None


def pytest_collection_modifyitems(items):
    for item in items:
        if marker := _dir_marker(item.path):
            item.add_marker(marker)
