"""This module provides an history tracker for the navigator module to track object's history.

Usage:
    - Adaptive Kalman Filter
       To track the history of covarience matrix and state vector of the object.

    - Smoothing Algorithm
       To track the history of the state vector of the object.
    
"""
import statistics
from queue import Queue
from typing import Any

__all__ = ["HistoryTracker"]


class HistoryTracker:
    """This class provides a history tracker for the navigator module to track object's history.

    The history tracker is used to track the history of the state vector and the covariance matrix of the object and
    other parameters.
    """

    def __init__(self, max_history: int) -> None:
        """Initialize the HistoryTracker class.

        Args:
            max_history (int): The maximum number of history to track.

        Returns:
            None
        """
        if max_history <= 0:
            raise ValueError("The maximum number of history to track must be positive.")
        self.max_history = max_history
        self.history = Queue(max_history)

    def add(self, data: Any) -> None:
        """Add the data to the history.

        Args:
            data (Any): The data to add to the history.

        Returns:
            None
        """
        # Remove the oldest data if the history is full
        if self.history.full():
            self.history.get()
        # Add the new data to the history
        self.history.put(data)

    def get(self) -> list[Any]:
        """Get the history.

        Returns:
            list[Any]: The history.
        """
        return list(self.history.queue)

    def clear(self) -> None:
        """Clear the history.

        Returns:
            None
        """
        self.history = Queue(self.max_history)

    def mean(self) -> Any:  # noqa
        """Get the mean of the history.

        Returns:
            Any: The mean of the history.
        """
        return statistics.mean(self.history.queue)

    def peek(self) -> Any:  # noqa
        """Peek the last element of the history.

        Returns:
            Any: The last element of the history.
        """
        return self.history.queue[-1]

    def __len__(self) -> int:
        """Get the length of the history.

        Returns:
            int: The length of the history.
        """
        return len(self.history.queue)

    def __getitem__(self, index: int) -> Any:  # noqa
        """Get the item at the specified index.

        Args:
            index (int): The index of the item to get.

        Returns:
            Any: The item at the specified index.
        """
        return self.history.queue[index]
