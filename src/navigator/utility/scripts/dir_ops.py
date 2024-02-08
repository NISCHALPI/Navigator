"""Epochify RINEX Directory .i.e convert RINEX files direcotry to Epoch Directory."""

from pathlib import Path

import click

from ...epoch.epoch_directory import EpochDirectory
from ..data_tools.standerd_directory import StanderdDirectory
from ..logger.logger import get_logger


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
    '-dp',
    '--data-path',
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
    '-p',
    '--process',
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


if __name__ == "__main__":
    main()
