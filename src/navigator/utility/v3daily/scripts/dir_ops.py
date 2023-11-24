"""Epochify RINEX Directory .i.e convert RINEX files direcotry to Epoch Directory."""

from pathlib import Path

import click

from ..data.epoch_directory import EpochDirectory
from ..data.standerd_directory import StanderdDirectory
from .logger import get_logger

# Define the logger.
logger = get_logger(__name__)
logger.info(f"Started logging for {__name__}...")


@click.group(invoke_without_command=True, no_args_is_help=True)
def main() -> None:
    """Epochify RINEX Directory .i.e convert RINEX files direcotry to Epoch Directory."""
    pass


@main.command()
@click.option(
    '-dp',
    '--data-path',
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    help="Path to RINEX directory containing RINEX files.",
)
def standerize(data_path: Path) -> None:
    """Standerize RINEX V3 files in RINEX directory. Delete all non-essential data."""
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
@click.option(
    '-dp',
    '--data-path',
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    help="Path to RINEX directory containing RINEX files.",
)
@click.option(
    '-ep',
    '--epoch-dir-path',
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    help="Path to directory to save epochified RINEX files.",
)
@click.option(
    '-t',
    '--triangulate',
    required=False,
    type=click.BOOL,
    default=True,
    help="Whether to triangulate the epoch and set results to epoch.position attribute.",
)
@click.option(
    '-p',
    '--process',
    required=False,
    type=click.IntRange(min=1),
    default=1,
    help="Number of processes to use for epochification.",
)
def epochify(
    data_path: Path, epoch_dir_path: Path, triangulate: bool, process: int
) -> None:
    """Epochify data contained in RINEX directory."""
    # Log the start of the epochification process.
    logger.info(f"Epochifying RINEX files in {data_path}...")
    logger.info(f"Target epoch directory: {epoch_dir_path}")

    # Epochify the RINEX files in the data directory.
    try:
        epoch_dir = EpochDirectory(
            directory_path=epoch_dir_path, triangulate=triangulate
        )
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


if __name__ == "__main__":
    main()
