"""This is the triangulate module for Navigator, serving as the primary module for user-end data processing for triangulation.

Design Pattern:
    - Builder: The `Triangulate` class is a builder class for the `AbstractTriangulationInterface` interface.

Interface Available:
    - AbstractTriangulationInterface (abc.ABC): An abstract interface for triangulation.
    - IterativeTriangulationInterface (AbstractTriangulationInterface): A concrete class for iterative triangulation.
    - UnscentedKalmanTriangulationInterface (AbstractTriangulationInterface): A concrete class for Unscented Kalman Filter triangulation.

Example Usage:
    >>> from navigator.core import Triangulate, IterativeTriangulationInterface
    >>> triangulator = Triangulate(interface=IterativeTriangulationInterface())
    >>> triangulator.process(obs=obs_epoch, nav_metadata=nav_metadata, obs_metadata=obs_metadata)

Note:
    Use the `Triangulate` class to process GNSS data for triangulation with respective interfaces.

See Also:
    - `navigator.core.satellite`: Handles satellite data processing.
    - `navigator.core.triangulate.itriangulate`: The Interface module for triangulation.

Todo:
    - Add other triangulation algorithms.
    - Migrate to a Rust backend for performance improvement.
"""

from .itriangulate.iterative.iterative_traingulation_interface import (
    IterativeTriangulationInterface,
)
from .itriangulate.kalman import UnscentedKalmanTriangulationInterface
from .triangulate import Triangulate
