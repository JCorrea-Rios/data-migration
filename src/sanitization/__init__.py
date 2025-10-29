"""
Company name sanitization and standardization.
"""

# Lazy imports to avoid loading heavy dependencies unnecessarily
# Users should import directly from the specific modules

__all__ = [
    'run',
    'run_with_batches',
    'GLiNERClassifier',
    'simple_sanitizer'
]

def __getattr__(name):
    """Lazy loading of heavy dependencies."""
    if name == 'run':
        from src.sanitization.company_name_sanitiser import run
        return run
    elif name == 'run_with_batches':
        from src.sanitization.company_name_sanitiser import run_with_batches
        return run_with_batches
    elif name == 'GLiNERClassifier':
        from src.sanitization.gliner_classifier import GLiNERClassifier
        return GLiNERClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")