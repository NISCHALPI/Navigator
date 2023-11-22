"""Implements a abstract base class for dispatchers.

Ideas:
Multiplethreaded dispatcher
Multiprocess dispatcher


Dipatch will be done using Idispatch interface.
"""


## TODO: Implement dispatcher

from abc import ABC

__all__ = ["AbstractDispatcher"]


class AbstractDispatcher(ABC):
    """Dispatch class."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        raise NotImplementedError
