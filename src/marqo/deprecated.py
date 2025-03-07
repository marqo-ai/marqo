import warnings
import functools


def deprecated(func):
    """This decorator marks functions as deprecated.
    It will result in a warning being emitted when the function is used.
    https://pypi.org/project/Deprecated/ might be a more powerful alternative. But we do not want to introduce a
    dependency for this simple feature.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper

