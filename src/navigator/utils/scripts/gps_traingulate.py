"""This module contains the script to triangulate the data from the RINEX files."""

import secrets
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ...core.triangulate import (
    IterativeTriangulationInterface,
    Triangulate,
)
from ...epoch import from_rinex_files
from ...logger.logger import get_logger
from ..igs_network.igs_network import IGSNetwork


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.pass_context
@click.option(
    "-v",
    "--verbose",
    required=False,
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
def main(
    ctx: click.Context,
    verbose: bool,
) -> None:
    """Triangulate the data from the RINEX files."""
    logger = get_logger(name=__name__, dummy=not verbose)
    logger.info("Starting triangulation!")

    # Ensure that the context object is dictionary-like
    ctx.ensure_object(dict)

    # Add the logger to the context object
    ctx.obj["logger"] = logger


@main.command(name="wls")
@click.pass_context
@click.option(
    "-nav",
    "--nav-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="The path to the RINEX navigation file",
)
@click.option(
    "-obs",
    "--obs-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="The path to the RINEX observation file",
)
@click.option(
    "-mode",
    "--mode",
    required=False,
    type=click.Choice(["single", "dual"]),
    default="single",
    help="The mode to triangulate the data",
)
@click.option(
    "-apply-tropo",
    "--apply-tropo",
    required=False,
    is_flag=True,
    default=False,
    help="Apply tropospheric correction to the data",
)
@click.option(
    "-apply-iono",
    "--apply-iono",
    required=False,
    is_flag=True,
    default=False,
    help="Apply ionospheric correction to the data",
)
@click.option(
    "-t",
    "--output-type",
    required=False,
    type=click.Choice(["csv", "html"]),
    default="html",
    help="The output file type",
)
@click.option(
    "--match-mode",
    required=False,
    type=click.Choice(["maxsv", "nearest"]),
    default="maxsv",
    help="The mode to pair the observations and navigation messages",
)
@click.option(
    "-i",
    "--ignore",
    required=False,
    type=click.INT,
    default=5,
    help="The epoch to ignore having less than this number of satellites",
)
@click.option(
    "-code-only",
    "--code-only",
    required=False,
    is_flag=True,
    default=False,
    help="Use only the code measurements for triangulation without the phase measurements",
)
@click.option(
    "-s",
    "--save",
    required=False,
    type=click.Path(exists=True, dir_okay=True, writable=True, path_type=Path),
    help="The path to save the data to",
    default=Path.cwd(),
)
@click.option(
    "-skip",
    "--skip",
    required=False,
    default=0,
    type=click.IntRange(min=0),
    help="The number of steps to skip in the data",
)
def wls(
    ctx: click.Context,
    nav_file: Path,
    obs_file: Path,
    mode: str,
    apply_tropo: bool,
    apply_iono: bool,
    output_type: str,
    match_mode: str,
    ignore: int,
    save: str,
    code_only: bool,
    skip: int,
) -> None:
    """Triangulate the data from the RINEX files using Weighted Least Squares (WLS)."""
    logger = ctx.obj["logger"]

    logger.info("Triangulating using Weighted Least Squares (WLS)!")

    logger.info("Epochifying the data!")
    # Epochify the data
    epoches = list(
        from_rinex_files(
            observation_file=obs_file,
            navigation_file=nav_file,
            mode=match_mode,
        )
    )
    logger.info(f"Found {len(epoches)} epoches!")

    N = skip
    # Filter out epoches with less than 5 satellites
    epoches = [epoch for epoch in epoches if len(epoch) > ignore]
    logger.info(f"Found {len(epoches)} epoches with more than {ignore} satellites!")

    # Triangulate the data
    triangulator = Triangulate(
        interface=IterativeTriangulationInterface(code_only=code_only)
    )

    triangulation_epoches = epoches[::N]

    # Change the profile of the epoches
    for epoch in triangulation_epoches:
        epoch.profile["apply_tropo"] = apply_tropo
        epoch.profile["apply_iono"] = apply_iono
        epoch.profile["mode"] = mode

    logger.info(
        f"""Triangulating {len(triangulation_epoches)} epoches with the following profile:
                
                - Apply Tropospheric Correction: {apply_tropo}
                - Apply Ionospheric Correction: {apply_iono}
                - Mode: {mode}
                - Match Mode: {match_mode}
                - Ignore: {ignore}
                - Skip: {N}
                """
    )
    df = triangulator.triangulate_time_series(epoches=triangulation_epoches)
    logger.info("Traingulation Completed! Saving the data!")

    # Convert to dataframe
    df = pd.DataFrame(df)
    # Save the data to the specified output type
    if output_type == "csv":
        df.to_csv(
            save / f"traingulation-wls-{secrets.token_urlsafe(nbytes=4)}.csv",
            index=False,
        )
    elif output_type == "html":
        df.to_html(
            save / f"traingulation-wls-{secrets.token_urlsafe(nbytes=4)}.html",
            index=False,
        )
    return


@main.command(name="igs-diff-plot")
@click.pass_context
@click.option(
    "-out-csv",
    "--output-csv",
    required=True,
    type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path),
    help="The path to the output csv file from the triangulation",
)
@click.option(
    "-st",
    "--station-name",
    required=True,
    type=click.STRING,
    help="The name of the station to plot",
)
@click.option(
    "-s",
    "--save-path",
    required=False,
    type=click.Path(exists=True, dir_okay=True, writable=True, path_type=Path),
    help="The path to save the plot to",
    default=Path.cwd(),
)
def igs_diff_plot(
    ctx: click.Context,
    output_csv: Path,
    station_name: str,
    save_path: Path,
) -> None:
    """Plot the IGS differences."""
    logger = ctx.obj["logger"]

    logger.info(
        "Plotting the differences between the IGS and the calculated positions!"
    )

    # Read the data
    df = pd.read_csv(output_csv)

    # Get the station coordinates
    network = IGSNetwork()
    if station_name not in network.stations.index:
        raise ValueError(f"Station {station_name} not found in the IGS network!")

    true_coords = network.get_xyz(
        station=station_name,
    )
    true_coords = pd.Series(
        {
            "x": true_coords[0],
            "y": true_coords[1],
            "z": true_coords[2],
        }
    )

    enu_error = Triangulate.enu_error(
        predicted=df,
        actual=true_coords,
    )

    # Plot the data
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Set seaborn theme
    sns.set_theme(
        style="darkgrid",
    )

    colnames = ["E_error", "N_error", "U_error"]

    for i, name in enumerate(colnames):
        sns.lineplot(
            x=df.index,
            y=enu_error[name],
            ax=ax[i],
            label=f"{name}",
        )

        ax[i].set_title(f"{name} vs Time")

    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path / f"enu-plot_{station_name}.png")

    # Plot the norm of the error
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    sns.lineplot(
        x=df.index,
        y=enu_error.apply(np.linalg.norm, axis=1),
        ax=ax,
        label="Total Error",
    )

    ax.set_title("Total Error vs Time")

    # Save the plot
    plt.savefig(save_path / f"error_plot_{station_name}.png")

    # Plot the histogram of the error
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    sns.histplot(
        x=enu_error.apply(np.linalg.norm, axis=1),
        ax=ax,
        kde=True,
    )

    ax.set_title("Histogram of the Total Error")

    # Save the plot
    plt.savefig(
        save_path / f"hist-error-plot_{station_name}.png",
    )

    return
