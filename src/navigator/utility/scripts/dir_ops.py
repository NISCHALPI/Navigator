"""Epochify RINEX Directory .i.e convert RINEX files direcotry to Epoch Directory."""

from pathlib import Path

import click
import os
from concurrent.futures import ProcessPoolExecutor

from ...epoch.epoch_directory import EpochDirectory
from ..logger.logger import get_logger
from ..rinex_data_tools.standerd_directory import StanderdDirectory


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
def main(ctx: click.Context, verbose: bool) -> None:
    """Epochify RINEX Directory .i.e convert RINEX files direcotry to Epoch Directory."""
    # Ensure the context object is dict
    ctx.ensure_object(dict)
    # Create a logger
    logger = get_logger(name=__name__, dummy=not verbose)

    # Add the logger to the context object
    ctx.obj["logger"] = logger

    # Log the start of the program
    logger.info("Starting Directory Operations process...")

    pass


@main.command()
@click.pass_context
@click.option(
    "-dp",
    "--data-path",
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    help="Path to RINEX directory containing RINEX files.",
)
def standerize(ctx: click.Context, data_path: Path) -> None:
    """Standerize RINEX V3 files in RINEX directory. Delete all non-essential data."""
    # Get the logger.
    logger = ctx.obj["logger"]
    # Log the start of the standerization process.
    logger.info(f"Standerizing RINEX files in {data_path}...")

    # Standarize the RINEX files in the data directory.
    try:
        StanderdDirectory(directory_path=data_path, clean=True)
    except Exception as e:
        logger.error(e)
        return

    # Log the end of the standerization process.
    logger.info("Standerization complete!")
    logger.info(f"Execute: '$: tree {data_path}' to view the standerized directory.")


@main.command()
@click.pass_context
@click.option(
    "-dp",
    "--data-path",
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    help="Path to RINEX directory containing RINEX files.",
)
@click.option(
    "-ep",
    "--epoch-dir-path",
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    help="Path to directory to save epochified RINEX files.",
)
@click.option(
    "-p",
    "--process",
    required=False,
    type=click.IntRange(min=1),
    default=1,
    help="Number of processes to use for epochification.",
)
def epochify(
    ctx: click.Context,
    data_path: Path,
    epoch_dir_path: Path,
    process: int,
) -> None:
    """Epochify data contained in RINEX directory."""
    # Get the logger.
    logger = ctx.obj["logger"]
    # Log the start of the epochification process.
    logger.info(f"Epochifying RINEX files in {data_path}...")
    logger.info(f"Target epoch directory: {epoch_dir_path}")

    # Epochify the RINEX files in the data directory.
    try:
        epoch_dir = EpochDirectory(directory_path=epoch_dir_path, logging=True)
    except ValueError as e:
        logger.info(
            "Epoch directory might not be empty. Please empty it and try again."
        )
        logger.error(e)
        return

    # Start the multiprocessing epochification process.
    logger.info(f"Epochifying RINEX files in {data_path} with {process} processes...")
    epoch_dir.populate(
        data_dir=data_path,
        process=process,
    )

    # Log the end of the epochification process.
    logger.info("Epochification complete!")


@main.command()
@click.pass_context
@click.option(
    "-f",
    "--file-path",
    required=True,
    type=click.Path(
        exists=True, readable=True, path_type=Path
    ),
    help="Path to Novtel Logs file.",
)
@click.option(
    "-s",
    "--save-path",
    required=True,
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path to save the extracted RINEX files.",
)
def novtel_to_rinex2(
    ctx: click.Context,
    file_path: Path,
    save_path: Path,
) -> None:
    """Extract RINEX V2 files from Novtel Logs directory."""
    # Get the logger.
    logger = ctx.obj["logger"]
    # Check the novtel app image path in environment variables.
    if "NOVTEL_APP_IMAGE" not in os.environ:
        raise ValueError("NOVTEL_APP_IMAGE environment variable not set.")
        return
    # Get the path to the novtel app image.
    novtel_app_image = Path(os.environ["NOVTEL_APP_IMAGE"])

    # Log the start of the epochification process.
    logger.info(f"Starting extracting RINEX V2 files from Novtel Logs in {file_path}...")
    logger.info(f"Target save directory: {save_path}")

    # Execute the conversion process.
    os.system(f"{novtel_app_image} -r2.1 -o {save_path} {file_path}")
   
    # Log the end of the epochification process.
    logger.info("Extraction complete!")


if __name__ == "__main__":
    main()
