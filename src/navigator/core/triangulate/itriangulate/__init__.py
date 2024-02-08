"""This is an interface module for the triangulation module.

Design Pattern:
    - Builder: The `Triangulate` class is a builder class for the `AbstractTriangulationInterface` interface.
    - Functional: Algorithms are implemented functionally rather than object-oriented.

Interface Available:
    - AbstractTriangulationInterface (abc.ABC): An abstract interface for triangulation.
    - IterativeTriangulationInterface (AbstractTriangulationInterface): A concrete class for iterative triangulation.
    - UnscentedKalmanTriangulationInterface (AbstractTriangulationInterface): A concrete class for Unscented Kalman Filter triangulation.

Example Usage:
    >>> from navigator.core import Triangulate, IterativeTriangulationInterface
    >>> triangulator = Triangulate(interface=IterativeTriangulationInterface())
    >>> triangulator.process(obs=obs_epoch, nav_metadata=nav_metadata, obs_metadata=obs_metadata)

Submodules:
    - `algos`: Contains the algorithms for triangulation.
    - `iterative`: Contains the iterative triangulation functional interface.
    - `kalman`: Contains the Unscented Kalman Filter triangulation functional interface.
    - `preprocess`: Contains the preprocessing functions for triangulation.

Note:
    Use the `Triangulate` class to process GNSS data for triangulation with respective interfaces.

See Also:
    - `navigator.core.satellite`: Handles satellite data processing.

Todo:
    - Add other triangulation algorithms.
    - Migrate to a Rust backend for performance improvement.
"""

from .itriangulate import Itriangulate
