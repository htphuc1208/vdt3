"""Readiness and integrity checks for the SpotLight Open RAN dataset."""

__all__ = ["spotlight_readiness"]


def spotlight_readiness(*args, **kwargs):
    from .readiness import spotlight_readiness as _spotlight_readiness

    return _spotlight_readiness(*args, **kwargs)
