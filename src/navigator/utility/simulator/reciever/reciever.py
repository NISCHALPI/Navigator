"""Simulate a receiver for a trajectory simulator.

This module provides the `Reciever` class which simulates a receiver capable of 
receiving signals from a satellite constellation based on a specified trajectory.

Classes:
    Reciever: Simulate a receiver for a trajectory simulator.
"""

import typing as tp

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from ....epoch import Epoch
from ..animation.time_series import timeseries_animation
from ..constellation.constellation import Constellation
from ..trajectory.trajectory import Trajectory

__all__ = ["Reciever"]


class Reciever:
    """Simulate a receiver that can receive a signal from a Constellation.

    Args:
        name (str): The name of the receiver.
        trajectory (Trajectory): The path of the receiver as a Trajectory object.
        constellation (Constellation): The constellation to receive signals from.

    Attributes:
        name (str): The name of the receiver.
        path (Trajectory): The path of the receiver.
        constellation (Constellation): The constellation to receive signals from.
    """

    START_EPOCH = pd.Timestamp("2021-01-01 00:00:00")

    def __init__(
        self, name: str, trajectory: Trajectory, constellation: Constellation
    ) -> None:
        """Initialize the Reciever class.

        Args:
            name (str): The name of the receiver.
            trajectory (Trajectory): The trajectory of the receiver as a Trajectory object.
            constellation (Constellation): The constellation to receive signals from.

        Returns:
            None
        """
        self.name = name
        self.path = trajectory
        self.constellation = constellation

    def record(self, time: float) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Record the signal received by the receiver at a given time.

        Args:
            time (float): The time at which the signal is received.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
                - observation_data (pd.DataFrame): DataFrame containing the observed range data for different signals.
                - navigation_data (pd.DataFrame): DataFrame containing the state of the constellation at the given time.
                - true_state (pd.Series): Series containing the true state (position and velocity) of the receiver.
        """
        # Get the current position and velocity of the receiver
        reciever_state = self.path(time=time)  # (6,)
        positions = reciever_state[:3]

        # Get the constellation state at the current time
        constellation_state = self.constellation.state(time)

        # Calculate the range between the receiver and the satellites
        range = np.linalg.norm(positions - constellation_state.values[:, :3], axis=1)

        # Create observation data dictionary
        observation_data = {
            Epoch.L1_CODE_ON: range,
            Epoch.L2_CODE_ON: range,
            Epoch.L1_PHASE_ON: range,
            Epoch.L2_PHASE_ON: range,
        }
        observation_data = pd.DataFrame.from_dict(observation_data, orient="columns")
        # Navigation data
        navigation_data = constellation_state

        # Apply same index to observation data
        observation_data.index = navigation_data.index

        # True state
        true_state = pd.Series(
            {
                "x": reciever_state[0],
                "y": reciever_state[1],
                "z": reciever_state[2],
                "vx": reciever_state[3],
                "vy": reciever_state[4],
                "vz": reciever_state[5],
            }
        )

        return observation_data, navigation_data, true_state

    def timeseries(
        self, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the epoch objects for the receiver at given times.

        Args:
            times (np.ndarray): Array of times at which the signal is received.(T,)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the observation data, navigation data and true states. (T,N,4), (T,N,6), (T,6)

        """
        obs_data = []
        nav_data = []
        true_states = []

        for time in times:
            obs, nav, true = self.record(time)
            obs_data.append(obs.values)
            nav_data.append(nav.values)
            true_states.append(true.values)

        return np.stack(obs_data), np.stack(nav_data), np.stack(true_states)

    def get_dummy_epoch(self, time: float) -> Epoch:
        """Get a dummy epoch object for the receiver at a given time.

        Args:
            time (float): The time at which the signal is received.

        Returns:
            Epoch: The dummy epoch object.
        """
        # Get the obs data, nav data and true state
        obs_data, nav_data, true_state = self.record(time)

        # Create the dummy epoch object
        epoch = Epoch(
            timestamp=self.START_EPOCH + pd.Timedelta(seconds=time),
            obs_data=obs_data,
            nav_data=nav_data,
            real_coord=true_state,
            obs_meta=pd.Series(),
            nav_meta=pd.Series(),
            trim=False,
            purify=False,
            station=f"SIM000_{self.name}",
        )

        # Activate the dummy profile
        epoch.profile = Epoch.DUMMY

        return epoch

    def animate(
        self,
        times: np.ndarray,
        range_dict: dict[str, np.ndarray],
        tracer_map: tp.Optional[dict[str, bool]] = None,
        text_map: tp.Optional[dict[str, bool]] = None,
    ) -> Figure:
        """Animate the receiver and the constellation.

        Args:
            times (np.ndarray): Array of times at which the signal is received.
            range_dict (dict[str, np.ndarray]): Dictionary containing the range data for different signals i.e xlim, ylim, zlim.
            tracer_map (dict[str, bool]): Dictionary containing the trace status for different signals.
            text_map (dict[str, str]): Dictionary containing the text for different signals.

        Returns:
            None
        """
        # Get the timeseries data
        obs_data, nav_data, true_states = self.timeseries(times)

        # Get the name of the constellation and the receiver
        names = [self.name] + list(self.constellation.trajectories.keys())

        # Initialize the tracer map and text map
        if tracer_map is None:
            tracer_map = {name: False for name in names}
        if text_map is None:
            text_map = {name: True for name in names}

        # Stack the position measurements as (T,1 + C, 3)
        positions = np.concatenate(
            [true_states[:, None, :3], nav_data[:, :, :3]], axis=1
        )
        # Generate random colors from continuous colormap
        colors = px.colors.qualitative.Light24

        # If the number of trajectories is greater than the number of colors, generate random colors
        if len(names) > len(colors):
            colors = px.colors.qualitative.Light24 * (len(names) // len(colors) + 1)

        color_map = dict(zip(names, colors))

        # Generate the animation
        fig = timeseries_animation(
            ts_data=positions,
            names=names,
            tracer_map=tracer_map,
            text_map=text_map,
            color_map=color_map,
        )

        # Update the layout
        fig.update_layout(
            title="Reciever Simulation",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="manual",
                aspectratio=dict(x=2, y=2, z=1),
                xaxis=dict(range=range_dict["x"]),
                yaxis=dict(range=range_dict["y"]),
                zaxis=dict(range=range_dict["z"]),
            ),
        )

        return fig

    def __call__(self, time: float) -> Epoch:
        """Get the epoch object for the receiver at a given time.

        Args:
            time (float): The time at which the signal is received.

        Returns:
            Epoch: The epoch object.
        """
        return self.get_dummy_epoch(time)
