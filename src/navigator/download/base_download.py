"""Implements base download class with idownload interface."""


import os
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

__all__ = ['Download']

from ..dispatch.base_dispatch import AbstractDispatcher
from .idownload import IDownload


class AbstractDownload(ABC):
    """_summary_.

    Args:
        ABC (_type_): _description_
    """

    def __init__(
        self,
        idownload: IDownload,
        save_path: str | None = None,
        logger: Logger | None = None,
        dispatcher: AbstractDispatcher | None = None,
    ) -> None:
        """_summary_.

        Args:
            idownload (Idownload): _description_
            save_path (str | None, optional): _description_. Defaults to None.
            logger (Logger | None, optional): _description_. Defaults to None.
            dispatcher (AbstractDispatcher | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            TypeError: _description_
            TypeError: _description_
            TypeError: _description_
        """
        # Path to save data
        if save_path is None:
            self.save_path = os.getcwd()
        else:
            # Check if path exists
            if not os.path.exists(save_path):
                raise ValueError(f"save_path must be valid path, not {save_path}")

            self.save_path = save_path

        # Check if idownload is valid
        if not issubclass(idownload.__class__, IDownload):
            raise TypeError(f"idownload must be Idownload, not {type(idownload)}")

        self.idownload = idownload

        # Verify logger is Logger
        if logger is not None and not isinstance(logger, Logger):
            raise TypeError(f"logger must be Logger, not {type(logger)}")
        self.logger = logger

        # Verify dispatcher is Dispatcher
        if dispatcher is not None and not issubclass(
            dispatcher.__class__, AbstractDispatcher
        ):
            raise TypeError(
                f'dispatcher must be subclass AbstractDispatcher, not {type(dispatcher)}'
            )
        self.dispatcher = dispatcher

        pass

    @abstractmethod
    def _download(self, *args, **kwargs) -> Any:  # noqa
        """_summary_.

        Args:
            *args (_type_): _description_
            **kwargs (_type_): _description_
        """
        # Execute the download method of the idownload object

        kwargs['save_path'] = self.save_path
        return self.idownload(*args, **kwargs)

    def __call__(self, *args, **kwds) -> Any:  # noqa
        return self._download(*args, **kwds)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.idownload},  {self.save_path})"


class Download(AbstractDownload):
    """Concrete implementation of AbstractDownload."""

    def _download(self, *args, **kwargs) -> Any:  # noqa
        return super()._download(*args, **kwargs)
