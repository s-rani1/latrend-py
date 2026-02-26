from __future__ import annotations


def plotnine_available() -> bool:
    try:
        import plotnine  # noqa: F401
    except Exception:
        return False
    return True


def choose_backend(backend: str | None) -> str:
    """
    Choose plotting backend.

    - If backend is None: prefer plotnine when installed, otherwise matplotlib.
    - If backend is set: validate and use it.
    """

    if backend is None:
        return "plotnine" if plotnine_available() else "matplotlib"

    backend = backend.lower().strip()
    if backend not in {"plotnine", "matplotlib"}:
        raise ValueError("backend must be one of: None, 'plotnine', 'matplotlib'")
    return backend


def require_plotnine():
    try:
        import plotnine as p9
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "plotnine backend requested but plotnine is not installed. "
            "Install with: pip install 'latrend[plot]' (or: pip install plotnine)."
        ) from e
    return p9

