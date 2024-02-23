from navigator.filters import (
    ExtendedKalmanFilter,
)
import numpy as np
from math import sqrt
from numpy.random import randn
import math
from filterpy.common import Q_discrete_white_noise
from numpy import eye, array, asarray
import numpy as np

# Define few function for the simulation


def HJacobian_at(x):
    """compute Jacobian of H matrix at x"""

    horiz_dist = x[0]
    altitude = x[2]
    denom = sqrt(horiz_dist**2 + altitude**2)
    return np.array([[horiz_dist / denom, 0.0, altitude / denom]])


def hx(x):
    """compute measurement for slant range that
    would correspond to state x.
    """

    return (x[0] ** 2 + x[2] ** 2) ** 0.5


class RadarSim:
    """Simulates the radar signal returns from an object
    flying at a constant altityude and velocity in 1D.
    """

    def __init__(self, dt, pos, vel, alt):
        self.pos = pos
        self.vel = vel
        self.alt = alt
        self.dt = dt

    def get_range(self):
        """Returns slant range to the object. Call once
        for each new measurement at dt time from last call.
        """

        # add some process noise to the system
        self.vel = self.vel + 0.1 * randn()
        self.alt = self.alt + 0.1 * randn()
        self.pos = self.pos + self.vel * self.dt

        # add measurement noise
        err = self.pos * 0.05 * randn()
        slant_dist = math.sqrt(self.pos**2 + self.alt**2)

        return slant_dist + err


def test_ekf_radar_sim():
    dt = 0.05
    rk = ExtendedKalmanFilter(dim_x=3, dim_y=1)
    radar = RadarSim(dt, pos=0.0, vel=100.0, alt=1000.0)

    # make an imperfect starting guess
    rk.x = array([radar.pos - 100, radar.vel + 100, radar.alt + 1000])

    # State Transition Matrix
    F = eye(3) + array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) * dt

    range_std = 5.0  # meters
    rk.R = np.diag([range_std**2])
    rk.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
    rk.Q[2, 2] = 0.1
    rk.P *= 50

    xz, track = [], []

    for i in range(int(20 / dt)):
        z = radar.get_range()
        track.append((radar.pos, radar.vel, radar.alt))
        rk.predict(F)
        rk.update(asarray([z]), hx, HJacobian_at, hx_kwargs={}, HJ_kwargs={})
        xz.append((rk._x[0], rk._x[1], rk._x[2]))

    xz = np.array(xz)
    track = np.array(track)

    # Assert the error in position is less than 15 meters
    assert np.allclose(xz[:, 0], track[:, 0], 15)
    # Assert the error in velocity is less than 5 m/s
    assert np.allclose(xz[:, 1], track[:, 1], 5)
    # Assert the error in altitude is less than 15 meters
    assert np.allclose(xz[:, 2], track[:, 2], 15)
