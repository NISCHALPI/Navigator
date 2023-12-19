from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from logging import NullHandler
from pathlib import Path


from fs.ftpfs import FTPFS

from ....utility.logger import get_logger
from ....utility.matcher import SP3Matcher, LegacySP3Matcher

from ..idownload import IDownload


class CddisIgsSP3(IDownload):
    pass
