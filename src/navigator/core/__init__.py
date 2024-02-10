"""This is the core module for Navigator, serving as the primary module for GNSS data processing.

Submodules:
    - `satellite`: Handles satellite data processing.
    - `triangulate`: Manages user-end data processing for triangulation.

Example Usage:
    >>> from navigator.core import Triangulate, IterativeTriangulationInterface
    >>> triangulator = Triangulate(interface=IterativeTriangulationInterface())
    >>> triangulator.process(obs=obs_epoch, nav_metadata=nav_metadata, obs_metadata=obs_metadata)

Note:
    This module acts as a super module for the `satellite` and `triangulate` modules. Refer to the respective modules for detailed information.

See Also:
    - `navigator.core.satellite`: Handles satellite data processing.
    - `navigator.core.triangulate`: Manages user-end data processing for triangulation.
"""

from .satellite import IGPSEphemeris, IGPSSp3, Satellite
from .triangulate import (
    IterativeTriangulationInterface,
    Triangulate,
    UnscentedKalmanTriangulationInterface,
)
from .triangulate.itriangulate.algos.slip_detection import GeometryFreeDetector
from .triangulate.itriangulate.algos.smoothing import (
    DivergenceFreeSmoother,
    HatchFilter,
    IonosphereFreeSmoother,
)
