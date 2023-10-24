"""Dynamic docstring for the navigator utility package."""

from functools import wraps

__all__ = ["dynamic_docstring"]


def dynamic_docstring(docstring: str) -> callable:
    """A decorator to set a dynamic docstring for functions and methods.

    Args:
        docstring (str): The dynamic docstring to set.

    Returns:
        function: The decorated function with the specified docstring.
    """

    def decorator(func):  # noqa
        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa
            return func(*args, **kwargs)

        wrapper.__doc__ = docstring
        return wrapper

    return decorator
