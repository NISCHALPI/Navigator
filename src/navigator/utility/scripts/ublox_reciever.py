"""Controller for the Ublox receiver."""

from pathlib import Path
from time import sleep

import click
import tqdm

from ..logger.logger import get_logger
from ..ublox.commands import NAV_PVT, RXM_RAWX, RXM_SFRBX
from ..ublox.profile import StreamingProfile

DEFAULT_COMMAND_WAIT_TIME = 0.01
DUMP_MESSAGE_WAIT_TIME = 2.0
WARMUP_TIME = 1


@click.command()
@click.option(
    "-d",
    "--device",
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    help="Path to the serial device file. Example: /dev/ttyACM0",
)
@click.option(
    "-b",
    "--baudrate",
    required=False,
    type=click.IntRange(min=1),
    default=115200,
    help="Baudrate of the serial connection. Default: 115200",
)
@click.option(
    "-rxm",
    "--rxm-log-path",
    required=True,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
    help="Path to the RXM-RAWX log file. Example: /tmp/rxm_rawx.ubx",
)
@click.option(
    "-pvt",
    "--pvt-log-path",
    required=True,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
    help="Path to the NAV-PVT log file. Example: /tmp/nav_pvt.ubx",
)
@click.option(
    "-t",
    "--time",
    required=False,
    type=click.FloatRange(min=1),
    default=60,
    help="Time to collect data. Default: 60 seconds",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose mode.",
    default=False,
)
def main(
    device: Path,
    baudrate: int,
    rxm_log_path: Path,
    pvt_log_path: Path,
    time: float,
    verbose: bool = False,
) -> None:
    """Collects data from the Ublox receiver.

    Args:
        ctx (Context): The context object.
        device (Path): The path to the serial device file.
        baudrate (int): The baudrate of the serial connection.
        rxm_log_path (Path): The path to the RXM-RAWX log file.
        pvt_log_path (Path): The path to the NAV-PVT log file.
        time (float): Time to collect data in seconds.
        verbose (bool): Enable verbose mode.
    """
    # If verbose mode is enabled, set the logger
    logger = get_logger("ublox", dummy=not verbose)

    # Create a StreamingProfile object
    profile = StreamingProfile(
        commands=[NAV_PVT(), RXM_RAWX(), RXM_SFRBX()],
        device=device,
        baudrate=baudrate,
        logger=logger,
        no_check=False,
    )

    # Open the log files
    with (
        rxm_log_path.open("wb") as rxm_log_file,
        pvt_log_path.open("wb") as pvt_log_file,
    ):
        # Start the data collection
        profile.start()

        # Create a time space for the data collection
        TIMESPACE = range(0, int(time), int(DUMP_MESSAGE_WAIT_TIME))

        # Create a progress bar for the data collection
        with tqdm.tqdm(total=len(TIMESPACE), desc="Collecting Data") as pbar:
            for _ in TIMESPACE:
                dump_periodic_messages(profile, rxm_log_file, pvt_log_file)
                pbar.update(1)

        profile.stop()

    return


def dump_periodic_messages(
    profile: StreamingProfile, rxm_log_file: Path, pvt_log_file: Path
) -> None:
    """Dumps the periodic messages to the log files.

    Args:
        profile (StreamingProfile): The streaming profile object.
        rxm_log_file (Path): The path to the RXM-RAWX log file.
        pvt_log_file (Path): The path to the NAV-PVT log file.
    """
    # WAIL for the command to be executed
    sleep(DUMP_MESSAGE_WAIT_TIME + DEFAULT_COMMAND_WAIT_TIME)

    # Get the data
    data = profile.collect()

    # Write the data to the log files
    for msg in data["RXM-RAWX"]:
        rxm_log_file.write(msg.serialize())

    for msg in data["NAV-PVT"]:
        pvt_log_file.write(msg.serialize())

    for msg in data["RXM-SFRBX"]:
        rxm_log_file.write(msg.serialize())

    sleep(DEFAULT_COMMAND_WAIT_TIME)

    return


if __name__ == "__main__":
    main()
