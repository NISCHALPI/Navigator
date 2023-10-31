"""This module defines the AbstractDirectory class.

The AbstractDirectory class is an abstract class to read data directory. It defines the interface to read data directory.

Example:
    To use the AbstractDirectory class, create a subclass that implements the clean method.

        class MyDirectory(AbstractDirectory):
            def clean(self):
                # implementation

Attributes:
    directory_path (str | Path): The path of the data directory.

Methods:
    clean() -> None: Clean the directory according to the rules of the subclass.
    print() -> None: Print the directory tree structure.
    __repr__() -> str: Representation of AbstractDirectory.
    __str__() -> str: String of AbstractDirectory.
    __iter__() -> Iterator: Iterate over the directory.
    __len__() -> int: Length of the directory.
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path

__all__ = ["AbstractDirectory"]

class AbstractDirectory(ABC):
    """Abstract Class to Read Data Directory. Define the interface to read data directory.

    Args:
        ABC (_type_): The ABC class.
    """
    
    def __init__(self, directory_path : str | Path) -> None:
        """Constructor of AbstractDirectory.

        Args:
            directory_path (str | Path): Path of Data Directory.
        """
        # Convert to Path object
        if isinstance(directory_path, str):
            directory_path = Path(directory_path)
            
        # Check if directory_path exists and is a directory and is readable
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory {directory_path} does not exist.")
        if not directory_path.is_dir():
            raise NotADirectoryError(f"{directory_path} is not a directory.")
        if not os.access(directory_path, os.R_OK):
            raise PermissionError(f"{directory_path} is not readable.")
        
        self.directory_path = directory_path
        
        
        
    @abstractmethod
    def clean(self) -> None:
        """Clean the directory according to the rules of the subclass."""
        pass
    
    
    def _print_directory_tree(
        self, root_path: str, indent: str = "", last: bool = True
    ) -> None:
        """Print the directory tree structure.

        Args:
            root_path (str): The root directory path.
            indent (str, optional): The indentation string. Defaults to "".
            last (bool, optional): True if the current directory is the last in its level.
        """
        print(indent + ("└─ " if last else "├─ ") + os.path.basename(root_path))
        if os.path.isdir(root_path):
            entries = os.listdir(root_path)
            entries.sort()
            for i, entry in enumerate(entries):
                entry_path = os.path.join(root_path, entry)
                is_last = i == len(entries) - 1
                self._print_directory_tree(
                    entry_path, indent + ("   " if last else "│  "), last=is_last
                )
    
    
    
    def print(self) -> None:
        """Print the directory tree structure."""
        self._print_directory_tree(self.directory_path)
    
    def __repr__(self) -> str:
        """Return a string representation of the AbstractDirectory object."""
        return f"{self.__class__.__name__}({self.directory_path})"
    
    
    
    def __str__(self) -> str:
        """Return a string representation of the AbstractDirectory object."""
        return self.__repr__()


    def __iter__(self): # noqa: ANN204
        """Return an iterator over the directory."""
        return iter(self.directory_path.iterdir())
    
    
    def __len__(self) -> int:
        """Return the number of entries in the directory."""
        return len(list(self.directory_path.iterdir()))
