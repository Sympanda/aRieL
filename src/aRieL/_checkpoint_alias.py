"""Let the published checkpoint import its original module path.

The zip was saved with policy class
``ariel_rl.agents.policies.full_set_isab_policy.FullSetISABPolicy``.
Importing ``aRieL`` registers an alias so that path loads this package.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys

_LEGACY = "ariel_rl"


class _CheckpointAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname != _LEGACY and not fullname.startswith(_LEGACY + "."):
            return None
        real = "aRieL" + fullname[len(_LEGACY):]
        real_spec = importlib.util.find_spec(real)
        if real_spec is None:
            return None
        is_package = real_spec.submodule_search_locations is not None
        spec = importlib.machinery.ModuleSpec(
            fullname,
            self,
            origin=real_spec.origin,
            is_package=is_package,
        )
        if is_package:
            spec.submodule_search_locations = list(real_spec.submodule_search_locations)
        spec._ariel_real = real  # type: ignore[attr-defined]
        return spec

    def create_module(self, spec):
        real = spec._ariel_real
        module = sys.modules.get(real)
        if module is None:
            module = importlib.import_module(real)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        return None


def install() -> None:
    if any(isinstance(finder, _CheckpointAlias) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _CheckpointAlias())
