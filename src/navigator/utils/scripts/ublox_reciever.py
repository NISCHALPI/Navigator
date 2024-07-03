"""Controller for the Ublox receiver."""

import shutil
from pathlib import Path
from time import sleep

import click
import pandas as pd
import pyubx2 as ubx
import tqdm

from ...logger.logger import get_logger
from ..ublox.commands import CFG_RATE, NAV_PVT, RXM_RAWX, RXM_SFRBX
from ..ublox.profile import StreamingProfile

DEFAULT_COMMAND_WAIT_TIME = 0.01
DUMP_MESSAGE_WAIT_TIME = 2.0
WARMUP_TIME = 1


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.pass_context
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose mode.",
    default=False,
)
def main(
    ctx: click.Context,
    verbose: bool = False,
) -> None:
    """Utility CLI interface for the Ublox receiver."""
    # If verbose mode is enabled, set the logger
    logger = get_logger("ublox", dummy=not verbose)

    # Add the logger to the context
    ctx.ensure_object(dict)

    # Add the logger to the context
    ctx.obj["logger"] = logger

    return


def dump_periodic_messages(
    profile: StreamingProfile, rxm_log_file: Path, pvt_log_file: Path
) -> int:
    """Dumps the periodic messages to the log files.

    Args:
        profile (StreamingProfile): The streaming profile object.
        rxm_log_file (Path): The path to the RXM-RAWX log file.
        pvt_log_file (Path): The path to the NAV-PVT log file.

    Returns:
        int: The number of messages collected and dumped.
    """
    # WAIL for the command to be executed
    sleep(DUMP_MESSAGE_WAIT_TIME + DEFAULT_COMMAND_WAIT_TIME)

    # Get the data
    num_msg, data = profile.collect()

    # Write the data to the log files
    for msg in data["RXM-RAWX"]:
        rxm_log_file.write(msg.serialize())

    for msg in data["NAV-PVT"]:
        pvt_log_file.write(msg.serialize())

    for msg in data["RXM-SFRBX"]:
        rxm_log_file.write(msg.serialize())

    sleep(DEFAULT_COMMAND_WAIT_TIME)

    return num_msg


@main.command(name="log")
@click.pass_context
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
    "-r",
    "--rate",
    required=False,
    type=click.FloatRange(min=1),
    default=1,
    help="Rate [Hz] to collect data. Default: 1 Hz",
)
def log(
    ctx: click.Context,
    device: Path,
    baudrate: int,
    rxm_log_path: Path,
    pvt_log_path: Path,
    time: float,
    rate: float,
) -> None:
    """Loggs the data from the Ublox receiver into UBX files."""
    # Get the logger from the context
    logger = ctx.obj["logger"]

    # Create a StreamingProfile object
    profile = StreamingProfile(
        commands=[NAV_PVT(), RXM_RAWX(), RXM_SFRBX()],
        device=device,
        baudrate=baudrate,
        logger=logger,
        no_check=False,
    )

    # Sent the rate to the receiver
    in_ms = int(1000 / rate)

    # Set the rate with the logger
    logger.info(f"Setting the rate to {rate} Hz.")

    # Send the rate to the receiver
    rateCmd = CFG_RATE().config_command(measRate=in_ms, navRate=1, timeRef=0)

    # Send the command to the receiver
    out = profile.controller.send_config_command(rateCmd, wait_for_ack=True)
    if out.identity != "ACK-ACK":
        click.echo("Failed to set the rate.")
        # Exit the program
        raise click.Abort()

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
        n = 0
        with tqdm.tqdm(total=len(TIMESPACE), desc=f"Collected {n} messages") as pbar:
            for _ in TIMESPACE:
                msg_collected = dump_periodic_messages(
                    profile, rxm_log_file, pvt_log_file
                )
                n += msg_collected
                # Update the progress bar
                pbar.set_description(f"Collected {n} messages")
                pbar.update(1)

        profile.stop()

    return


@main.command(name="pvt-to-csv")
@click.pass_context
@click.option(
    "-p",
    "--pvt-log-path",
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    help="Path to the NAV-PVT log file. Example: /tmp/nav_pvt.ubx",
)
@click.option(
    "-c",
    "--csv-path",
    required=True,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
    help="Path to the CSV file. Example: /tmp/nav_pvt.csv",
)
def pvt_to_csv(ctx: click.Context, pvt_log_path: Path, csv_path: Path) -> None:
    """Converts the NAV-PVT log file to a CSV file."""
    # Get the logger from the context
    logger = ctx.obj["logger"]

    # Open the log file
    logger.info(f"Opening the log file: {pvt_log_path}")

    # Create a NAV-PVT object
    nav_pvt_cmd = NAV_PVT()

    with pvt_log_path.open("rb") as pvt_log_file:
        # Create a UBX reader
        reader = ubx.UBXReader(
            datastream=pvt_log_file,
            protfilter=ubx.UBX_PROTOCOL,
        )

        # Loop through the messages
        data = []

        for raw, parsed in reader:
            # If message is NAV-PVT
            if parsed.identity == nav_pvt_cmd:
                data.append(nav_pvt_cmd.parse_ubx_message(parsed))

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Convert the time column to timestamp
    timecols = ["year", "month", "day", "hour", "min", "second"]

    df["timestamp"] = pd.to_datetime(
        df[timecols].rename(
            columns={
                "year": "year",
                "month": "month",
                "day": "day",
                "hour": "hour",
                "min": "minute",
                "seconds": "second",
            }
        )
    )

    # Drop the time columns
    df.drop(columns=timecols, inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

    return


@main.command(name="rxm-to-rinex")
@click.pass_context
@click.option(
    "-r",
    "--rxm-log-path",
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    help="Path to the RXM-RAWX log file. Example: /tmp/rxm_rawx.ubx",
)
def rxm_to_rinex(ctx: click.Context, rxm_log_path: Path) -> None:
    """Converts the RXM-RAWX log file to a RINEX file."""
    logger = ctx.obj["logger"]

    logger.info(f"Opening the log file: {rxm_log_path}")

    command = "convbin"

    if shutil.which(command) is None:
        click.echo(
            f"Conversion command {command} not found on PATH. Please install RTKLIB and add it to PATH."
        )
        click.Abort()
        return

    try:
        logger.info(f"Converting {rxm_log_path} to RINEX format using {command}.")
        shutil.run([command, str(rxm_log_path)])
        logger.info("Conversion completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during conversion: {e}")


if __name__ == "__main__":
    main()
