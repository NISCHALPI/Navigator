"""Includes various body trajectory for simulating moving body.

A trajectory is a path that a moving body follows through space as a function of time. This module includes various types of trajectories that can be used to simulate the movement of a body in a 3D space.

Classes:
    - EightTrajectory: Represents a trajectory where the body moves along a figure-eight path.
    - StationaryTrajectory: Represents a stationary trajectory where the body remains at a fixed position.
    - SatelliteLikeTrajectory: Represents a trajectory where the body moves like a satellite in an elliptical orbit.
    - EllipticalTrajectory: Represents an elliptical trajectory where the body moves along an elliptical path.
    - KepelerianSatellite: Represents a satellite trajectory using keplerian elements.
    - gps_factory: Factory method to create a trajectory based on the type of trajectory.
"""

from .gps import gps_factory, KepelerianSatellite
from .eight import EightTrajctory
from .stationary import StationaryTrajectory
from .orbit import SatelliteLikeTrajectory
from .ellipsoidal import EllipticalTrajectory
