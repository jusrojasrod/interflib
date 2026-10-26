"""Deprecation helpers for renamed arguments and attributes of the public API."""
import functools
import warnings


def warn_deprecated(message: str, stacklevel: int = 3) -> None:
    """Issue a :class:`DeprecationWarning` attributed to the caller of the public function."""
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def renamed_arguments(**renames):
    """Accept old keyword names during a transition, with a :class:`DeprecationWarning`.

    ``renamed_arguments(old="new")`` maps the keyword ``old`` onto ``new``; passing both
    is an error.
    """
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"{func.__qualname__}() got both {old!r} and {new!r}")
                    warn_deprecated(f"{func.__qualname__}(): argument {old!r} is deprecated; use {new!r}")
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)
        return wrapper
    return decorate
