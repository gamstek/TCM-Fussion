"""Backward-compatible five-image alias for the generic multi-image API."""
from .multi_view import bind_config, inspect_sources, run_multi_view

# Old callers keep working; the implementation no longer enforces five images.
run_five_view = run_multi_view

__all__ = ['bind_config', 'inspect_sources', 'run_multi_view', 'run_five_view']
