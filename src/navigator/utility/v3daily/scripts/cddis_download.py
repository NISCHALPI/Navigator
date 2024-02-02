"""Download rinex files from CCIDS using curlftpfs."""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import click
import tqdm

from ....download.idownload.rinex.nasa_cddis import NasaCddisV3
from ....download.idownload.sp3.ccdis_igs_sp3 import NasaCddisIgsSp3

# ------------------------------------------ Global Variables ------------------------------
# USER EMAIL
USER_EMAIL = "navigator@navigator.github.com"


# ------------------------------------------ Click Commands ------------------------------------------------
@click.group(invoke_without_command=True, no_args_is_help=True)
@click.pass_context
@click.option(
    "-e",
    "--email",
    required=False,
    default=USER_EMAIL,
    help=f"Email to login to CCIDS. Default: {USER_EMAIL}",
)
@click.option(
    "-t",
    "--threads",
    required=False,
    default=1,
    type=click.IntRange(1),
    help="Number of threads to use for downloading",
)
@click.option(
    "-v",
    "--verbose",
    required=False,
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
@click.version_option(version="1.0.0")
def main(ctx: click.Context, email: str, threads: int, verbose: bool) -> None:
    """Download RINEX files from CCIDS using without mounting the server."""
    # Ensure the context object is dict
    ctx.ensure_object(dict)
    # Add verbose to the context
    ctx.obj["verbose"] = verbose
    # Create the Downloader
    ctx.obj["downloader"] = NasaCddisV3(email=email, logging=verbose, threads=threads)

    return


@main.command()
@click.pass_context
@click.option(
    "-p",
    "--path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, writable=True, path_type=Path),
    help="Path to save the files",
)
@click.option(
    "-y", "--year", required=True, type=click.INT, help="Year to download RINEX files"
)
@click.option(
    "-d",
    "--day",
    required=True,
    type=click.IntRange(1, 366),
    help="Day of year to download RINEX files",
)
@click.option(
    "-s",
    "--samples",
    required=False,
    type=click.INT,
    default=-1,
    help="Number of samples to download",
)
@click.option(
    "-m",
    "--match",
    required=False,
    type=click.STRING,
    default=None,
    help="IGS station code to match",
)
def daily(
    ctx: click.Context, path: Path, year: int, day: int, samples: int, match: str
) -> None:
    """Download RINEX files for the given year and day."""
    # Download the files from downloader
    downloader: NasaCddisV3 = ctx.obj["downloader"]

    # Download the files for the given year and day
    downloader.download(
        year=year,
        day=day,
        save_path=path,
        num_files=samples,
        no_pbar=False,
        match_string=match,
    )
    return


@main.command()
@click.pass_context
@click.option(
    "-p",
    "--path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, writable=True, path_type=Path),
    help="Path to save the files",
)
@click.option(
    "-y", "--year", required=True, type=click.INT, help="Year to download RINEX files"
)
@click.option(
    "-s",
    "--samples",
    required=False,
    type=click.INT,
    default=-1,
    help="Number of samples to download",
)
@click.option(
    "--process",
    required=False,
    type=click.INT,
    default=1,
    help="Number of processes to use for downloading",
)
@click.option(
    "-m",
    "--match",
    required=False,
    type=click.STRING,
    default=None,
    help="IGS station code to match",
)
def yearly(
    ctx: click.Context, path: Path, year: int, samples: int, process: int, match: str
) -> None:
    """Download RINEX files for the given year."""
    # Get the range of days
    start_day, end_day = 1, 366

    # If current year, get the range of days till yesterday
    if year == datetime.now().year:
        start_day = 1
        end_day = datetime.now().timetuple().tm_yday - 1

    # Get the downloader
    downloader: NasaCddisV3 = ctx.obj["downloader"]

    # Disable the logging in multiprocessing
    downloader.logger.disabled = True

    # Download the files for the given year and day using multiprocessing
    with tqdm.tqdm(total=end_day - start_day + 1, desc="Downloading Files!") as pbar:
        with ProcessPoolExecutor(max_workers=process) as executor:
            for day in range(start_day, end_day + 1):
                future = executor.submit(
                    downloader.download,
                    year=year,
                    day=day,
                    save_path=path,
                    num_files=samples,
                    no_pbar=True,
                    match_string=match,
                )

                # Add the callback to update the progress bar
                future.add_done_callback(lambda p: pbar.update(1))  # noqa : ARG005

        # Wait for all the processes to finish
        executor.shutdown(wait=True)

    return


@main.command()
@click.pass_context
@click.option(
    "-p",
    "--path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, writable=True, path_type=Path),
    help="Path to save the files",
)
@click.option(
    "-y",
    "--year",
    required=True,
    type=click.INT,
    help="Year to download RINEX SP3 files",
)
@click.option(
    "-m",
    "--month",
    required=True,
    type=click.IntRange(1, 12),
    help="Month of year to download RINEX SP3 files",
)
@click.option(
    "-d",
    "--day",
    required=True,
    type=click.IntRange(1, 31),
    help="Day of month to download RINEX SP3 files",
)
def sp3(ctx: click.Context, path: Path, year: int, month: int, day: int) -> None:
    """Download SP3 files for the given year, month and day."""
    # Create the downloader
    downloader = NasaCddisIgsSp3(logging=ctx.obj["verbose"])

    # Create a datetime object for the given year, month and day
    dt = datetime(year, month, day)

    # Download the SP3 files
    downloader.download_from_datetime(
        time=dt,
        save_dir=path,
    )
    return


# ------------------------------------------ END ----------------------------------------------------------------------------------

# ------------------------------------------ Main Function ---------------------------------
if __name__ == "__main__":
    main()
# ------------------------------------------ END ---------------------------------
