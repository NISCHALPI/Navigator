"""This module contains the IDownload interface."""


import typing as tp
from abc import ABC, abstractmethod
from typing import Any

__all__ = ["IDownload"]


class IDownload(ABC):
    """_summary_.

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, features: str) -> None:
        """_summary_.

        Args:
            features (str): _description_
        """
        # Feature type
        self._features = features if features else "NoneType"
        super().__init__()

    @abstractmethod
    def _download(self, *args, **kwargs) -> tp.Any:  # noqa
        """_summary_.

        Returns:
            tp.Any: _description_
        """
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:  # noqa
        return self._download(*args, **kwds)

    def __repr__(self) -> str:  # noqa
        return f"{self.__class__.__name__}({self._features})"
