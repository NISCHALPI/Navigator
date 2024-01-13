"""Implements an abstract base class for dispatchers.

Ideas:
- Multiplethreaded dispatcher
- Multiprocess dispatcher

Dispatch will be done using the Idispatch interface.

"""

## TODO: Implement dispatcher

from abc import ABC

__all__ = ["AbstractDispatcher"]


class AbstractDispatcher(ABC):
    """Dispatch class.

    Attributes:
        None

    Methods:
        __init__: Constructor

    """

    def __init__(self) -> None:
        """Constructor.

        Raises:
            NotImplementedError: This is an abstract class and should not be instantiated directly.

        """
        super().__init__()
        raise NotImplementedError
