"""Commands for the u-blox module which provides methods for recieving data from the u-blox receiver.

Classes:
    - BaseCommand: Base class for all u-blox commands.
    - NAV-PVT : Command to get the navigation position velocity time solution.
    - RXM-RAWX: Command to get the raw Mesurement data range and carrier phase.    

"""

import typing as tp
from abc import ABC, abstractmethod
import pyubx2 as ubx
import pandas as pd


__all__ = ["BaseCommand", "NAV_POSLLH", "NAV_POSECEF", "NAV_PVT", "RXM_RAWX"]


class BaseCommand(ABC):
    """Base class for all u-blox commands."""

    def __init__(self, msg_cls: str, msg_id: str) -> None:
        """Constructor for the BaseCommand class."""
        self.msg_cls = msg_cls
        self.msg_id = msg_id
        return

    def __repr__(self) -> str:
        """Return a string representation of the command."""
        return f"{self.msg_cls}-{self.msg_id}"

    @abstractmethod
    def config_command(self) -> ubx.UBXMessage:
        """Return the configuration command message rate."""
        pass

    @abstractmethod
    def parse_ubx_message(self, ubx_message: ubx.UBXMessage) -> pd.Series:
        """Return the parsed payload data as a pandas Series."""
        pass

    @abstractmethod
    def units(self) -> tp.Dict[str, str]:
        """Return the units for the parsed payload data."""
        pass

    def __eq__(self, value: str) -> bool:
        return str(self) == value


class NAV_POSLLH(BaseCommand):
    def __init__(self) -> None:
        """Constructor for the NAV-POSLLH class."""
        return super().__init__("NAV", "POSLLH")

    def config_command(self, on: str = "USB") -> ubx.UBXMessage:
        """Return the configuration command message rate.

        Args:
            on (str): The message rate configuration on channel. Defaults to "USB".

        Returns:
            UBXMessage: The configuration command message rate.
        """
        return ubx.UBXMessage.config_set(
            layers=ubx.SET_LAYER_RAM,
            transaction=ubx.TXN_NONE,
            cfgData=[(f"CFG_MSGOUT_UBX_{self.msg_cls}_{self.msg_id}_{on}", 1)],
        )

    def parse_ubx_message(self, ubx_message: ubx.UBXMessage) -> pd.Series:
        """Return the parsed payload data as a pandas Series.

        Args:
            ubx_message (UBXMessage): The UBX message to parse.

        Returns:
            Series: The parsed payload data.
        """
        return pd.Series(
            {
                "iTOW": ubx_message.iTOW / 1000,
                "lon": ubx_message.lon,
                "lat": ubx_message.lat,
                "height": ubx_message.height / 1000,
                "hMSL": ubx_message.hMSL / 1000,
                "hAcc": ubx_message.hAcc / 1000,
                "vAcc": ubx_message.vAcc / 1000,
            }
        )

    def units(self) -> tp.Dict[str, str]:
        """Return the units for the parsed payload data.

        Returns:
            Dict[str, str]: The units for the parsed payload data.
        """
        return {
            "iTOW": "s",
            "lon": "deg",
            "lat": "deg",
            "height": "m",
            "hMSL": "m",
            "hAcc": "m",
            "vAcc": "m",
        }


class NAV_POSECEF(BaseCommand):
    def __init__(self) -> None:
        """Constructor for the NAV-POSECEF class."""
        return super().__init__("NAV", "POSECEF")

    def config_command(self, on: str = "USB") -> ubx.UBXMessage:
        """Return the configuration command message rate.

        Args:
            on (str): The message rate configuration on channel. Defaults to "USB".

        Returns:
            UBXMessage: The configuration command message rate.
        """
        return ubx.UBXMessage.config_set(
            layers=ubx.SET_LAYER_RAM,
            transaction=ubx.TXN_NONE,
            cfgData=[(f"CFG_MSGOUT_UBX_{self.msg_cls}_{self.msg_id}_{on}", 1)],
        )

    def parse_ubx_message(self, ubx_message: ubx.UBXMessage) -> pd.Series:
        """Return the parsed payload data as a pandas Series.

        Args:
            ubx_message (UBXMessage): The UBX message to parse.

        Returns:
            Series: The parsed payload data.
        """
        return pd.Series(
            {
                "iTOW": ubx_message.iTOW,
                "ecefX": ubx_message.ecefX / 100,
                "ecefY": ubx_message.ecefY / 100,
                "ecefZ": ubx_message.ecefZ / 100,
                "pAcc": ubx_message.pAcc / 100,
            }
        )

    def units(self) -> tp.Dict[str, str]:
        """Return the units for the parsed payload data.

        Returns:
            Dict[str, str]: The units for the parsed payload data.
        """
        return {
            "iTOW": "ms",
            "ecefX": "m",
            "ecefY": "m",
            "ecefZ": "m",
            "pAcc": "m",
        }


class NAV_PVT(BaseCommand):
    def __init__(self) -> None:
        """Constructor for the NAV-PVT class."""
        return super().__init__("NAV", "PVT")

    def config_command(self, on: str = "USB") -> ubx.UBXMessage:
        """Return the configuration command message rate.

        Args:
            on (str): The message rate configuration on channel. Defaults to "USB".

        Returns:
            UBXMessage: The configuration command message rate.
        """
        return ubx.UBXMessage.config_set(
            layers=ubx.SET_LAYER_RAM,
            transaction=ubx.TXN_NONE,
            cfgData=[(f"CFG_MSGOUT_UBX_{self.msg_cls}_{self.msg_id}_{on}", 1)],
        )

    def parse_ubx_message(self, ubx_message: ubx.UBXMessage) -> pd.Series:
        """Return the parsed payload data as a pandas Series.

        Args:
            ubx_message (UBXMessage): The UBX message to parse.

        Returns:
            Series: The parsed payload data.
        """
        return pd.Series(
            {
                "iTOW": ubx_message.iTOW,
                "year": ubx_message.year,
                "month": ubx_message.month,
                "day": ubx_message.day,
                "hour": ubx_message.hour,
                "min": ubx_message.min,
                "second": ubx_message.second,
                "validDate": ubx_message.validDate,
                "validTime": ubx_message.validTime,
                "fullyResolved": ubx_message.fullyResolved,
                "validMag": ubx_message.validMag,
                "tAcc": ubx_message.tAcc,
                "nano": ubx_message.nano,
                "fixType": ubx_message.fixType,
                "gnssFixOk": ubx_message.gnssFixOk,
                "difSoln": ubx_message.difSoln,
                "psmState": ubx_message.psmState,
                "headVehValid": ubx_message.headVehValid,
                "carrSoln": ubx_message.carrSoln,
                "confirmedAvai": ubx_message.confirmedAvai,
                "confirmedDate": ubx_message.confirmedDate,
                "confirmedTime": ubx_message.confirmedTime,
                "numSV": ubx_message.numSV,
                "lon": ubx_message.lon,
                "lat": ubx_message.lat,
                "height": ubx_message.height / 1000,
                "hMSL": ubx_message.hMSL / 1000,
                "hAcc": ubx_message.hAcc / 1000,
                "vAcc": ubx_message.vAcc / 1000,
                "velN": ubx_message.velN / 1000,
                "velE": ubx_message.velE / 1000,
                "velD": ubx_message.velD / 1000,
                "gSpeed": ubx_message.gSpeed / 1000,
                "headMot": ubx_message.headMot,
                "sAcc": ubx_message.sAcc,
                "headAcc": ubx_message.headAcc,
                "pDOP": ubx_message.pDOP,
                "invalidLlh": ubx_message.invalidLlh,
                "lastCorrectionAge": ubx_message.lastCorrectionAge,
                "reserved0": ubx_message.reserved0,
                "headVeh": ubx_message.headVeh,
                "magDec": ubx_message.magDec,
                "magAcc": ubx_message.magAcc,
            }
        )

    def units(self) -> tp.Dict[str, str]:
        """Return the units for the parsed payload data.

        Returns:
            Dict[str, str]: The units for the parsed payload data.
        """
        return {
            "iTOW": "ms",
            "year": "year",
            "month": "month",
            "day": "day",
            "hour": "hour",
            "min": "min",
            "second": "s",
            "validDate": "bool",
            "validTime": "bool",
            "fullyResolved": "bool",
            "validMag": "bool",
            "tAcc": "ns",
            "nano": "ns",
            "fixType": "enum",
            "gnssFixOk": "bool",
            "difSoln": "bool",
            "psmState": "enum",
            "headVehValid": "bool",
            "carrSoln": "enum",
            "confirmedAvai": "bool",
            "confirmedDate": "year",
            "confirmedTime": "s",
            "numSV": "count",
            "lon": "deg",
            "lat": "deg",
            "height": "m",
            "hMSL": "m",
            "hAcc": "m",
            "vAcc": "m",
            "velN": "m/s",
            "velE": "m/s",
            "velD": "m/s",
            "gSpeed": "m/s",
            "headMot": "deg",  # Check this
            "sAcc": "m/s",
            "headAcc": "deg",  # Check this
            "pDOP": "unitless",
            "invalidLlh": "bool",
            "lastCorrectionAge": "s",
            "reserved0": "reserved",
            "headVeh": "deg",
            "magDec": "deg",
            "magAcc": "deg",
        }


class RXM_RAWX(BaseCommand):
    def __init__(self) -> None:
        """Constructor for the RXM-RAWX class."""
        super().__init__("RXM", "RAWX")

    def config_command(self, on: str = "USB") -> ubx.UBXMessage:
        """Return the configuration command message rate.

        Args:
            on (str): The message rate configuration on channel. Defaults to "USB".

        Returns:
            UBXMessage: The configuration command message rate.
        """
        return ubx.UBXMessage.config_set(
            layers=ubx.SET_LAYER_RAM,
            transaction=ubx.TXN_NONE,
            cfgData=[(f"CFG_MSGOUT_UBX_{self.msg_cls}_{self.msg_id}_{on}", 1)],
        )

    def parse_ubx_message(self, ubx_message: ubx.UBXMessage) -> pd.DataFrame:
        """Return the parsed payload data as a pandas DataFrame.

        Args:
            ubx_message (UBXMessage): The UBX message to parse.

        Returns:
            DataFrame: The parsed payload data.
        """
        # Create a list to store the data
        data = []

        # Loop through the Mesurements
        rcvTow = ubx_message.rcvTow
        rcvWeek = ubx_message.week
        leapS = ubx_message.leapS
        numMeas = ubx_message.numMeas
        leapSec = ubx_message.leapSec
        clkReset = ubx_message.clkReset

        # Loop through the numMes
        for id in range(1, numMeas + 1):
            id = str(id).zfill(2)
            subData = {}
            subData["prMes"] = getattr(ubx_message, f"prMes_{id}")
            subData["cpMes"] = getattr(ubx_message, f"cpMes_{id}")
            subData["doMes"] = getattr(ubx_message, f"doMes_{id}")
            subData["gnssId"] = getattr(ubx_message, f"gnssId_{id}")
            subData["svId"] = getattr(ubx_message, f"svId_{id}")
            subData["sigId"] = getattr(ubx_message, f"sigId_{id}")
            subData["freqId"] = getattr(ubx_message, f"freqId_{id}")
            subData["locktime"] = getattr(ubx_message, f"locktime_{id}")
            subData["cno"] = getattr(ubx_message, f"cno_{id}")
            subData["prStd"] = getattr(ubx_message, f"prStd_{id}")
            subData["cpStd"] = getattr(ubx_message, f"cpStd_{id}")
            subData["doStd"] = getattr(ubx_message, f"doStd_{id}")
            subData["prValid"] = getattr(ubx_message, f"prValid_{id}")
            subData["halfCyc"] = getattr(ubx_message, f"halfCyc_{id}")
            subData["subHalfCyc"] = getattr(ubx_message, f"subHalfCyc_{id}")

            # Add some of these to the data list
            subData["rcvTow"] = rcvTow
            subData["rcvWeek"] = rcvWeek
            subData["clockReset"] = clkReset

            data.append(subData)

        return pd.DataFrame(data)

    def units(self) -> tp.Dict[str, str]:
        """Return the units for the parsed payload data.

        Returns:
            Dict[str, str]: The units for the parsed payload data.
        """
        return {
            "rcvTow": "s",
            "rcvWeek": "week",
            "leapS": "s",
            "numMes": "count",
            "recStat": "enum",
            "leapSec": "s",
            "clkReset": "bool",
            "version": "version",
            "reserved0": "reserved",
            "prMes": "m",
            "cpMes": "cycles",
            "doMes": "m",
            "gnssId": "enum",
            "svId": "count",
            "sigId": "enum",
            "freqId": "enum",
            "locktime": "ms",
            "cno": "dBHz",
            "prStdev": "m",
            "prStd": "cycles",
            "cpStdev": "m",
            "cpStd": "cycles",
            "doStdev": "m",
            "doStd": "cycles",
            "trkStat": "enum",
            "prValid": "bool",
            "halfCyc": "cycles",
            "subHalfCyc": "cycles",
            "reserved": "reserved",
        }
