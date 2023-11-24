"""Makes a logger for the scripts in this directory."""
import logging
import os

# Check if the LOGLEVEL variable is set. If it is, use that as the log level.
# Otherwise, use the default log level of INFO.
if "LOGLEVEL" in os.environ and os.environ["LOGLEVEL"] in logging._nameToLevel:
    loglevel = logging._nameToLevel[os.environ["LOGLEVEL"]]
else:
    loglevel = logging.INFO

# Set the logging format.
logging.basicConfig(level=loglevel, format="%(asctime)s - %(levelname)s - %(message)s")


# Define the logger.
def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name."""
    return logging.getLogger(name)
