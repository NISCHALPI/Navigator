"""Module for simulating and visualizing satellite constellations.

This module provides functionality to simulate the trajectories of satellite constellations and visualize their movement in 3D space using Plotly.

Classes:
    Constellation: A class representing a satellite constellation, which contains multiple trajectories.
    
Functions:
    get_constellation: A factory function to create different types of satellite constellations based on the given name.

Usage:
    To create a satellite constellation and simulate its movement:
    
    ```python
    from navigator.constellation import Constellation, get_constellation
    
    # Create a default constellation with 8 stationary trajectories
    constellation = get_constellation()
    
    # Simulate the constellation movement for a given time array
    times = np.linspace(0, 100, 1000)  # Example time array
    fig = constellation.animate(times, range_dict={"x": [-50, 50], "y": [-50, 50], "z": [-50, 50]})
    
    # Display the animation
    fig.show()
    ```
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..animation.time_series import timeseries_animation
from ..trajectory.trajectory import Trajectory

__all__ = ["Constellation"]


class Constellation:
    """A class representing a satellite constellation, which contains multiple trajectories.

    This class provides functionality to manage a satellite constellation consisting of multiple trajectories. Each trajectory represents the movement path of a satellite within the constellation.

    Attributes:
        trajectories (dict[str, Trajectory]): A dictionary containing the name of each trajectory and its corresponding Trajectory object.

    Note:
        - The trajectory dictionary should contain the name of each trajectory as the key and its corresponding Trajectory object as the value.
        - Each trajectory's name should be a unique string identifier.
    """

    def __init__(self, trajectories: dict[str, Trajectory]) -> None:
        """Initialize the Constellation object with the given trajectories.

        Args:
            trajectories (dict[str, Trajectory]): A dictionary containing the name of each trajectory and its corresponding Trajectory object.

        Raises:
            ValueError: If the trajectories dictionary is empty.
            ValueError: If the trajectories has unique keys.
            ValueError: If all the values in the dictionary are not Trajectory objects.
        """
        # If empty dictionary is passed
        if len(trajectories) == 0:
            raise ValueError("Trajectories dictionary should not be empty.")

        # Check if the trajectories has unique keys
        if len(trajectories) != len(set(trajectories.keys())):
            raise ValueError("Trajectories should have unique keys.")

        # Check if all the values in the dictionary are Trajectory objects
        if not all(
            isinstance(trajectory, Trajectory) for trajectory in trajectories.values()
        ):
            raise ValueError(
                "All the values in the dictionary should be Trajectory objects."
            )

        self.trajectories = trajectories

    def state(self, time: float) -> pd.DataFrame:
        """Get the state of all the trajectories at the given time.

        Args:
            time (float): The time at which the state of the trajectories is required.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the state of all the trajectories at the given time.

        Note:
            - State is defined as [x, y, z, vx, vy, vz]
        """
        # Get the state of all the trajectories at the given time
        state = {}
        for name, trajectory in self.trajectories.items():
            pos_and_velocity = trajectory(time=time)
            state[name] = {
                "x": pos_and_velocity[0],
                "y": pos_and_velocity[1],
                "z": pos_and_velocity[2],
                "vx": pos_and_velocity[3],
                "vy": pos_and_velocity[4],
                "vz": pos_and_velocity[5],
            }

        return pd.DataFrame.from_dict(state, orient="index")

    def get_timeseries(self, times: np.ndarray) -> np.ndarray:
        """Get the time series data for the constellation.

        Args:
            times (np.ndarray): The time array for which the constellation is to be simulated. (N,)

        Returns:
            np.ndarray: The time series data for the constellation. (T, N, 6)

        Note:
            - N is the number of trajectories in the constellation.
            - T is the number of time steps in the time array.
            - The last dimension contains the position and velocity data for each trajectory.
        """
        # Initialize the time series data
        time_series = np.zeros(
            (len(times), len(self.trajectories), 6), dtype=np.float64
        )

        # Get the state of the constellation at each time step
        for i, t in enumerate(times):
            time_series[i] = self.state(t).to_numpy(dtype=np.float64)

        return time_series

    def animate(
        self,
        times: np.ndarray,
        range_dict: dict,
        tracer: bool = True,
        no_text: bool = False,
    ) -> go.Figure:
        """This function simulates the constellation for a given time.

        Args:
            times (np.ndarray): The time array for which the constellation is to be simulated.
            range_dict (dict): A dictionary containing the range of the X, Y, Z axis.
            tracer (bool): A boolean value to enable/disable the tracer. Default is True.
            no_text (bool): A boolean value to enable/disable the text labels. Default is False.

        Returns:
            go.Figure: A plotly figure object containing the animation of the constellation.
        """
        # Get the timeseries data for the constellation
        names = list(self.trajectories.keys())
        time_series = self.get_timeseries(times)

        # Generate random colors from continuous colormap
        colors = px.colors.qualitative.Light24

        # If the number of trajectories is greater than the number of colors, generate random colors
        if len(names) > len(colors):
            colors = px.colors.qualitative.Light24 * (len(names) // len(colors) + 1)

        # Get the timeseries data for the constellation
        fig = timeseries_animation(
            time_series,
            names,
            {name: tracer for name in names},
            {name: not no_text for name in names},
            {name: colors[i] for i, name in enumerate(names)},
        )

        # Update the layout
        fig.update_layout(
            title="Constellation Simulation",
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

    def __call__(self, time: float) -> pd.DataFrame:
        """Get the state of all the trajectories at the given time.

        Args:
            time (float): The time at which the state of the trajectories is required.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the state of all the trajectories at the given time.
        """
        return self.state(time)

    def __repr__(self) -> str:
        """Get the string representation of the Constellation object.

        Returns:
            str: String representation of the Constellation object.
        """
        return f"Constellation(num_trajectories={len(self.trajectories)})"

    def __mul__(self, scale: float) -> "Constellation":
        """Scales the constellation by a factor.

        Args:
            scale (float): The scale factor.

        Returns:
            Constellation: A new Constellation object scaled by the given factor.
        """
        return Constellation(
            {name: trajectory * scale for name, trajectory in self.trajectories.items()}
        )
