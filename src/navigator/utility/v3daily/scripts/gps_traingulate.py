"""This module contains the script to triangulate the data from the RINEX files."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pandas as pd
import georinex as gr

from ....parse import IParseGPSNav, IParseGPSObs, Parser
from ....satlib.triangulate import GPSIterativeTriangulationInterface, Triangulate
from ...epoch import Epoch
from ...logger.logger import get_logger


@click.group(no_args_is_help=True, invoke_without_command=True)
@click.pass_context
@click.option(
    "-v",
    "--verbose",
    required=False,
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
def main(ctx: click.Context, verbose: bool) -> None:
    """Triangulate the data from the RINEX files."""
    # Ensure that the ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Set the logger
    logger = get_logger(name=__name__, dummy=not verbose)

    # Set the logger in the context
    logger.info("Setting the logger in the context!")
    ctx.obj["logger"] = logger

    # Create a GPS triangulator object
    logger.info("Creating a GPS triangulator object")
    ctx.obj["triangulator"] = Triangulate(
        interface=GPSIterativeTriangulationInterface(),
    )

    logger.info("Creating a GPS navigation parser")
    ctx.obj["nav_parser"] = Parser(iparser=IParseGPSNav())

    logger.info("Creating a GPS observation parser")
    ctx.obj["obs_parser"] = Parser(iparser=IParseGPSObs())


@main.command(name="all-epochs")
@click.pass_context
@click.option(
    "-o",
    "--obs",
    type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "-n",
    "--nav",
    type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "-t",
    "--threads",
    type=click.IntRange(min=1),
    required=False,
    default=1,
    help="Number of threads to use for triangulation",
)
def triangulate_all_epochs(
    ctx: click.Context, obs: Path, nav: Path, threads: int
) -> None:
    """Triangulate all the epochs in the RINEX files and returns the mean position of the reciver."""
    # Get the logger
    logger = ctx.obj["logger"]

    # Validate the input
    logger.info(f"Observation file: {obs}")
    logger.info(f"Navigation file: {nav}")

    # Epochify the data
    logger.info("Epochifying the data. This may take a while if the file is large.")
    epochs = list(Epoch.epochify(obs=obs, nav=nav, mode="maxsv"))
    # Get the metadata
    nav_meta = gr.rinexheader(nav)
    obs_meta = gr.rinexheader(obs)

    logger.info("Available epochs:")
    # log the epochs
    for epoch in epochs:
        logger.info(f"Epoch: {epoch}")

    # Log epoch Summary
    logger.info(f"Number of epochs with 4> SV: {len(epochs)}")

    # Start traingulation
    logger.info(f"Starting triangulation for each epoch with {threads} threads!")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = executor.map(
            ctx.obj["triangulator"].__call__,
            epochs,
            [obs_meta] * len(epochs),
            [nav_meta] * len(epochs),
        )
        # Log the results
        executor.shutdown(wait=True)

    # Create a list of results
    results = list(results)
    # Create a dataframe
    df = pd.DataFrame(results)

    # Save the dataframe to a csv file
    logger.info(f"Saving the results to a csv file at : {Path.cwd() / 'results.csv'}")

    df.to_csv(Path.cwd() / "results.csv", index=False)

    # Print the mean of the results
    print(df.mean(axis=0))

    return


if __name__ == "__main__":
    main()
