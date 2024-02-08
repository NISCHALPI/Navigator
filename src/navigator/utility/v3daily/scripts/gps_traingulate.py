"""This module contains the script to triangulate the data from the RINEX files."""

import secrets
from pathlib import Path

import click
import pandas as pd
import tqdm

from ....core.triangulate import (
    IterativeTriangulationInterface,
    Triangulate,
    UnscentedKalmanTriangulationInterface,
)
from ....epoch import Epoch
from ...logger.logger import get_logger


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
    "-t",
    "--output-type",
    required=False,
    type=click.Choice(["csv", "html"]),
    default="html",
    help="The output file type",
)
@click.option(
    "--mode",
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
    "-s",
    "--save",
    required=False,
    type=click.Path(exists=True, dir_okay=True, writable=True, path_type=Path),
    help="The path to save the data to",
    default=Path.cwd(),
)
def wls(
    ctx: click.Context,
    nav_file: Path,
    obs_file: Path,
    output_type: str,
    mode: str,
    ignore: int,
    save: str,
) -> None:
    """Triangulate the data from the RINEX files using Weighted Least Squares (WLS)."""
    logger = ctx.obj["logger"]

    logger.info("Triangulating using Weighted Least Squares (WLS)!")

    logger.info("Epochifying the data!")
    # Epochify the data
    epoches = list(Epoch.epochify(obs=obs_file, nav=nav_file, mode=mode))
    logger.info(f"Found {len(epoches)} epoches!")

    # Filter out epoches with less than 5 satellites
    epoches = [epoch for epoch in epoches if len(epoch) > ignore]
    logger.info(f"Found {len(epoches)} epoches with more than {ignore} satellites!")

    # Triangulate the data
    triangulator = Triangulate(interface=IterativeTriangulationInterface())

    df = []
    # Triangulate the data
    with tqdm.tqdm(total=len(epoches)) as pbar:
        for epoch in epoches:
            df.append(
                triangulator(
                    epoch,
                    prior=df[-1] if len(df) > 0 else None,
                )
            )
            pbar.update(1)

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


@main.command(name="ukf")
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
    "-t",
    "--output-type",
    required=False,
    type=click.Choice(["csv", "html"]),
    default="html",
    help="The output file type",
)
@click.option(
    "--mode",
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
    "--ukf-dt",
    required=False,
    type=float,
    default=30.0,
    help="The time step for the UKF triangulation",
)
@click.option(
    "--ukf-sigma-r",
    required=False,
    type=float,
    default=6,
    help="The error in the range measurement",
)
@click.option(
    "--ukf-sigma-q",
    required=False,
    type=float,
    default=0.01,
    help="The process noise for the UKF triangulation",
)
@click.option(
    "--ukf-Sf",
    required=False,
    type=float,
    default=36,
    help="The white noise spectral density for the random walk clock velocity error. Defaults to 36.",
)
@click.option(
    "--ukf-Sg",
    required=False,
    type=float,
    default=0.01,
    help="The white noise spectral density for the random walk clock drift error. Defaults to 36.",
)
@click.option(
    "-s",
    "--save",
    required=False,
    type=click.Path(exists=True, dir_okay=True, writable=True, path_type=Path),
    help="The path to save the data to",
    default=Path.cwd(),
)
def ukf(
    ctx: click.Context,
    nav_file: Path,
    obs_file: Path,
    output_type: str,
    mode: str,
    ignore: int,
    ukf_dt: float,
    ukf_sigma_r: float,
    ukf_sigma_q: float,
    ukf_sf: float,
    ukf_sg: float,
    save: Path,
) -> None:
    """Triangulate the data from the RINEX files using Unscented Kalman Filter (UKF)."""
    logger = ctx.obj["logger"]

    logger.info("Triangulating using Unscented Kalman Filter (UKF)!")

    # Epochify the data
    epoches = list(Epoch.epochify(obs=obs_file, nav=nav_file, mode=mode))
    logger.info(f"Found {len(epoches)} epoches!")

    # Filter out epoches with less than 5 satellites
    epoches = [epoch for epoch in epoches if len(epoch) > ignore]
    logger.info(f"Found {len(epoches)} epoches with more than {ignore} satellites!")

    # Triangulate the data
    triangulator = Triangulate(
        interface=UnscentedKalmanTriangulationInterface(
            num_satellite=ignore,
            dt=ukf_dt,
            sigma_r=ukf_sigma_r,
            sigma_q=ukf_sigma_q,
            S_f=ukf_sf,
            S_g=ukf_sg,
            saver=False,
        )
    )

    df = []
    # Triangulate the data
    with tqdm.tqdm(total=len(epoches)) as pbar:
        for epoch in epoches:
            df.append(
                triangulator(
                    epoch,
                )
            )
            pbar.update(1)

    logger.info("Traingulation Completed! Saving the data!")
    # Convert to dataframe
    df = pd.DataFrame(df)
    # Save the data to the specified output type
    if output_type == "csv":
        df.to_csv(
            save / f"traingulation-ukf-{secrets.token_urlsafe(nbytes=4)}.csv",
            index=False,
        )
    elif output_type == "html":
        df.to_html(
            save / f"traingulation-ukf-{secrets.token_urlsafe(nbytes=4)}.html",
            index=False,
        )

    return
