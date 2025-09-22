"""Runtime type-check configuration for GCPNet modules."""

from __future__ import annotations

import os
from typing import Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., object])


def _noop_decorator(*_args, **_kwargs):
    def decorator(func: _F) -> _F:
        return func

    return decorator


def _noop_function(func: _F) -> _F:
    return func


_enable_checks = os.environ.get("GCPNET_ENABLE_RUNTIME_TYPECHECK", "0")
_enable_checks = str(_enable_checks).lower() in {"1", "true", "yes"}

try:  # pragma: no cover - optional dependency
    from jaxtyping import jaxtyped as _jaxtyped
except ModuleNotFoundError:  # pragma: no cover - jaxtyping not installed
    _jaxtyped = None

try:  # pragma: no cover - optional dependency
    from beartype import beartype as _beartype
except ModuleNotFoundError:  # pragma: no cover - beartype not installed
    _beartype = None

if _enable_checks and _jaxtyped is not None and _beartype is not None:
    jaxtyped = _jaxtyped
    typechecker = _beartype
else:  # default: no runtime type checking to keep hot path lean
    jaxtyped = _noop_decorator  # type: ignore[assignment]
    typechecker = _noop_function  # type: ignore[assignment]

__all__ = ["jaxtyped", "typechecker"]
