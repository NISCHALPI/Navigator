"""Download module for Navigator."""
from .base_download import Download
from .idownload import AusGovDownload
from .idownload.rinex.nasa_cddis import NasaCddisV3
from .idownload.sp3.ccdis_igs_sp3 import NasaCddisIgsSp3
