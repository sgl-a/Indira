"""
Guards that packaging metadata matches the code.

Two drifts have bitten this project before and are invisible until someone
installs from a clean checkout:

1. A registered provider pointing at a module that no longer exists.
2. A provider importing a package that `pyproject.toml` never declares — so
   `pip install -e ".[...]"` succeeds and the provider then fails to load.

Both are checked statically: no provider module is imported and no network or
installed distribution is required, so this passes on a bare checkout.
"""

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

from src.core.registry import _CATEGORIES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_DIR = PROJECT_ROOT / "src" / "providers"

# Third-party module name → the distribution that provides it. Update this when
# a provider starts importing something new; an unmapped import fails the test
# rather than passing silently.
MODULE_TO_DIST = {
    "mlx_whisper": "mlx-whisper",
    "mlx_audio": "mlx-audio",
    "mlx": "mlx-audio",      # ships as a dependency of mlx-audio
    "chromadb": "chromadb",
    "ollama": "ollama",
    "numpy": "numpy",
    "soundfile": "soundfile",
    "sounddevice": "sounddevice",
    "httpx": "httpx",
    "yaml": "pyyaml",
    "rich": "rich",
}


def _pyproject() -> dict:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _declared_distributions() -> set[str]:
    """Every distribution named in dependencies or any extra, normalized."""
    project = _pyproject()["project"]
    specs = list(project.get("dependencies", []))
    for extra_specs in project.get("optional-dependencies", {}).values():
        specs.extend(extra_specs)

    names = set()
    for spec in specs:
        # "chromadb>=1.5" → "chromadb"; PEP 503 normalization
        name = spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _provider_modules() -> list[Path]:
    return sorted(PROVIDERS_DIR.rglob("*_provider.py"))


def _imported_top_level_modules(path: Path) -> set[str]:
    """Top-level module names imported by a file, including lazy imports
    inside functions (which is how providers import their heavy deps)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_registry_paths_resolve_to_real_modules():
    """Every registered provider points at a module that exists.

    Catches a provider being deleted or renamed without updating the registry —
    otherwise the failure only surfaces when someone selects it in config.
    """
    for category, (registry, _default) in _CATEGORIES.items():
        for name, dotted_path in registry.items():
            module_path = dotted_path.rsplit(".", 1)[0]
            spec = importlib.util.find_spec(module_path)
            assert spec is not None, (
                f"{category} provider '{name}' points at {module_path}, "
                f"which does not exist"
            )


def test_default_providers_are_registered():
    """Each category's fallback default is actually in its registry."""
    for category, (registry, default_name) in _CATEGORIES.items():
        assert default_name in registry, (
            f"{category} default '{default_name}' is not registered"
        )


@pytest.mark.parametrize(
    "provider_path", _provider_modules(), ids=lambda p: p.stem
)
def test_provider_imports_are_declared(provider_path: Path):
    """Every third-party package a provider imports is declared in pyproject.

    This is the check that would have caught `whisper = ["openai-whisper"]`
    while the provider imported `mlx_whisper`.
    """
    declared = _declared_distributions()
    stdlib = sys.stdlib_module_names

    for module in _imported_top_level_modules(provider_path):
        if module in stdlib or module == "src" or module == "__future__":
            continue

        assert module in MODULE_TO_DIST, (
            f"{provider_path.name} imports '{module}', which is not in "
            f"MODULE_TO_DIST. Add it there and make sure pyproject.toml "
            f"declares the distribution that provides it."
        )

        dist = MODULE_TO_DIST[module]
        assert dist in declared, (
            f"{provider_path.name} imports '{module}' (from '{dist}'), "
            f"but pyproject.toml declares no such dependency. A clean "
            f"`pip install -e \".[...]\"` would not install it."
        )
