# tests/conftest.py
"""Root-level conftest.

Pre-imports real torch / transformers / safetensors / numpy
if they are installed, so that phase-specific conftest files (Phase 1~7)
skip mock injection. This prevents mock modules from polluting sys.modules
and breaking TTS tests that require real PyTorch.

On environments where these packages are NOT installed, the pre-import is
skipped and phase conftest files inject mocks as before.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys


# Modules that have real installations in this environment.
# Note: diffusers is NOT included here because real diffusers imports trigger
# pipeline module loading that requires specific transformers classes, causing
# collection errors. Phase conftests inject mock diffusers instead.
_REAL_MODULES = ("torch", "transformers", "safetensors", "numpy", "imageio", "soundfile")


def _is_real(name: str) -> bool:
    """Check if sys.modules[name] is a real module (has __file__ attribute)."""
    mod = sys.modules.get(name)
    return mod is not None and getattr(mod, "__file__", None) is not None


def _preimport_real_modules() -> None:
    """Pre-import real heavy modules if installed.

    This runs at conftest load time (before any phase conftest), ensuring
    that ``"torch" in sys.modules`` is True when phase conftests check it,
    causing them to skip mock injection.
    """
    for name in _REAL_MODULES:
        if name in sys.modules:
            if _is_real(name):
                continue  # Already real
            # It's a mock, remove it so we can import the real one
            for key in list(sys.modules.keys()):
                if key == name or key.startswith(name + "."):
                    sys.modules.pop(key, None)
        # Try direct import
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            pass


_preimport_real_modules()
