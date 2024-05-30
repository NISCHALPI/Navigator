"""Animation functions for time series data.

This module provides functions to create animations of time series data, particularly useful for visualizing trajectories in three-dimensional space.

Functions:
    - timeseries_animation: Creates a timeseries animation of the constellation.

Classes:
    None

"""

import numpy as np
import plotly.graph_objects as go

__all__ = ["timeseries_animation"]


def timeseries_animation(
    ts_data: np.ndarray,
    names: list[str],
    tracer_map: dict[str, bool],
    text_map: dict[str, bool],
    color_map: dict[str, str] = None,
) -> go.Figure:
    """Creates a timeseries animation of the constellation.

    Args:
        ts_data (np.ndarray): The time series data for the constellation. (T, N, 6)
        names (list[str]): A list of trajectory names. (N,)
        tracer_map (dict[str, bool]): A dictionary mapping trajectory names to tracer boolean values. (N,)
        text_map (dict[str, bool]): A dictionary mapping trajectory names to text boolean values. (N,)
        color_map (dict[str, str], optional): A dictionary mapping trajectory names to color values. (N,)

    Returns:
        go.Figure: A plotly figure object containing the animation of the constellation.
    """
    # Initialize the figure
    fig = go.Figure()

    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="manual",
            aspectratio=dict(x=2, y=2, z=1),
        )
    )

    # Add the initial positions and labels of the towers
    for i in range(len(ts_data[0])):
        fig.add_trace(
            go.Scatter3d(
                x=[ts_data[0, i, 0]],
                y=[ts_data[0, i, 1]],
                z=[ts_data[0, i, 2]],
                mode="markers+text" if text_map[names[i]] else "markers",
                marker=dict(size=5, color=color_map[names[i]]),
                text=[names[i]],
                textposition="top center",
                name=names[i],
                showlegend=True,
            )
        )

        if tracer_map[names[i]]:
            fig.add_trace(
                go.Scatter3d(
                    x=ts_data[:1, i, 0].flatten(),
                    y=ts_data[:1, i, 1].flatten(),
                    z=ts_data[:1, i, 2].flatten(),
                    mode="lines",
                    line=dict(color=color_map[names[i]], width=2),
                    name=f"{names[i]}_tracer",
                    showlegend=False,
                )
            )

    # Add the animation frames for both moving markers and growing traces
    frames = []
    for i in range(len(ts_data)):
        frame_data = []
        tracer_frame_data = []
        for j, name in enumerate(names):
            frame_data.append(
                go.Scatter3d(
                    x=[ts_data[i, j, 0]],
                    y=[ts_data[i, j, 1]],
                    z=[ts_data[i, j, 2]],
                    mode="markers+text" if text_map[name] else "markers",
                    marker=dict(size=5, color=color_map[name]),
                    text=[name],
                    textposition="top center",
                    name=name,
                    showlegend=True,
                )
            )

            if tracer_map[name]:
                tracer_frame_data.append(
                    go.Scatter3d(
                        x=ts_data[: i + 1, j, 0].flatten(),
                        y=ts_data[: i + 1, j, 1].flatten(),
                        z=ts_data[: i + 1, j, 2].flatten(),
                        mode="lines",
                        line=dict(color=color_map[name], width=2),
                        name=f"{name}_tracer",
                        showlegend=False,
                    )
                )

        frames.append(
            go.Frame(
                data=frame_data + tracer_frame_data,
                name=f"frame_{i}",
                traces=list(
                    range(len(frame_data) + len(tracer_frame_data)),
                ),
            )
        )

    # Add the frames for moving markers and tracer lines to the figure
    fig.frames = frames

    # Update the layout to include animation buttons
    fig.update_layout(
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="Stop",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
            )
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [f"frame_{i}"],
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(i),
                        "method": "animate",
                    }
                    for i in range(len(ts_data))
                ],
                "active": 0,
                "y": 0,
                "x": 0,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"b": 10, "t": 10},
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Time:",
                    "visible": True,
                    "xanchor": "right",
                },
                "len": 0.9,
                "tickcolor": "white",
            }
        ],
    )

    return fig
