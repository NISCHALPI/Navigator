"""GNSS Satellite Trajectory Simulator.

This module defines a Python class, GNSSsatellite, for simulating the trajectory of a GNSS satellite in space. 
The simulation is based on Keplerian orbital elements, and the class provides a method to calculate the coordinates of the satellite at a given time.


Attributes:
    __all__ (List[str]): A list of strings specifying the names of the public objects exported by the module.

Sample Almanac Data:
    Sample almanac data for initializing GNSS satellites.

Classes:
    KepelerianSatellite: A class representing a GNSS satellite trajectory simulator based on Keplerian orbital elements.

Methods:
    KepelerianSatellite.get_coords: Returns the coordinates of the satellite at a given time.

    KepelerianSatellite.get_pos_at_time: Calculates the position of the satellite at a given time.

    KepelerianSatellite.get_velocity_at_time: Calculates the velocity of the satellite at a given time.

Constants:
    ALMANAC_STATIC (Dict[str, Dict[str, Union[int, float]]]): Static almanac data for initializing GNSS satellites.
"""

import typing as tp
from pathlib import Path

from numpy import ndarray

from ....core.satellite.iephm.sv.tools.ephemeris_algos import ephm_to_coord_gps
from ....parse.iparse.yuma_alm.iparse_yuma_alm import IParseYumaAlm
from .trajectory import Trajectory

__all__ = ["KepelerianSatellite", "from_almanac"]


class KepelerianSatellite(Trajectory):
    """Simulates the trajectory of a GNSS satellite in space.

    Methods:
        get_coords: Returns the coordinates of the satellite at a given time.

        get_pos_at_time: Calculates the position of the satellite at a given time.

        get_velocity_at_time: Calculates the velocity of the satellite at a given time.

    Attributes:
        prn (str): PRN of the satellite.
        sqrtA (float): Square root of the semi-major axis.
        eccentricty (float): Eccentricity of the orbit.
        time_of_almanac (float): Time of almanac in seconds.
        inclination (float): Inclination of the orbit in radians.
        right_ascension (float): Right ascension at the week in radians.
        rate_of_right_ascension (float): Rate of right ascension in radians per second.
        argument_of_perigee (float): Argument of perigee in radians.
        mean_anomaly (float): Mean anomaly in radians.
        week (int): Week number.
        health (int): Health status of the satellite.
        af0 (float): Clock correction coefficient af0 in seconds.
        af1 (float): Clock correction coefficient af1 in seconds per second.

    Note:
        The parameters are based on the Yuma almanac format.
    """

    def __init__(
        self,
        prn: str,
        sqrtA: float,
        eccentricty: float,
        time_of_almanac: float,
        inclination: float,
        right_ascension: float,
        rate_of_right_ascension: float,
        argument_of_perigee: float,
        mean_anomaly: float,
        week: int,
        health: int,
        af0: float,
        af1: float,
    ) -> None:
        """Initializes the Keplerian satellite with the given parameters from the almanac.

        Args:
            prn (str): PRN of the satellite.
            sqrtA (float): Square root of the semi-major axis.
            eccentricty (float): Eccentricity of the orbit.
            time_of_almanac (float): Time of almanac in seconds.
            inclination (float): Inclination of the orbit in radians.
            right_ascension (float): Right ascension at the week in radians.
            rate_of_right_ascension (float): Rate of right ascension in radians per second.
            argument_of_perigee (float): Argument of perigee in radians.
            mean_anomaly (float): Mean anomaly in radians.
            week (int): Week number.
            health (int): Health status of the satellite.
            af0 (float): Clock correction coefficient af0 in seconds.
            af1 (float): Clock correction coefficient af1 in seconds per second.

        Returns:
            None

        Note:
            The parameters are based on the Yuma almanac format.
        """
        # Initialize the Keplerian satellite with the given parameters
        self.prn = prn
        self.sqrtA = sqrtA
        self.eccentricty = eccentricty
        self.time_of_almanac = time_of_almanac
        self.inclination = inclination
        self.right_ascension = right_ascension
        self.rate_of_right_ascension = rate_of_right_ascension
        self.argument_of_perigee = argument_of_perigee
        self.mean_anomaly = mean_anomaly
        self.week = week
        self.health = health
        self.af0 = af0
        self.af1 = af1

        super().__init__(prn)

    def get_coords(self, t: float) -> tuple:
        """Calculate the coordinates (x, y, z) of the satellite at a given time t.

        Parameters:
        - t: Time elapsed from the initial time (in seconds)

        Returns:
        - Tuple (x, y, z) representing the coordinates of the satellite
        """
        # Correct the time for the satellite clock error (af0 and af1) to GPS time
        t_corrected = (
            self.time_of_almanac + t - (self.af0 + self.af1 * t)
        )  # Account for satellite clock error

        return ephm_to_coord_gps(
            t=t_corrected,
            toe=self.time_of_almanac,
            sqrt_a=self.sqrtA,
            e=self.eccentricty,
            M_0=self.mean_anomaly,
            w=self.argument_of_perigee,
            i_0=self.inclination,
            omega_0=self.right_ascension,
            omega_dot=self.rate_of_right_ascension,
            c_ic=0,
            c_is=0,
            c_rc=0,
            c_rs=0,
            c_uc=0,
            c_us=0,
            delta_n=0,
            i_dot=0,
        )

    def get_pos_at_time(self, time: float) -> ndarray:
        """Calculates the position of the satellite at a given time.

        Args:
            time (float): Time at which the position is calculated.

        Returns:
            np.ndarray: Position of the satellite [x, y, z].
        """
        return self.get_coords(time)

    def get_velocity_at_time(self, time: float) -> ndarray:
        """Calculates the velocity of the satellite at a given time.

        Args:
            time (float): Time at which the velocity is calculated.

        Returns:
            np.ndarray: Velocity of the satellite [vx, vy, vz].

        Note:
            The velocity is calculated using the finite difference method.
        """
        # Calculate the velocity using finite difference method
        delta_t = 1e-7

        # Calculate the position at time t
        pos_t = self.get_coords(time)
        pos_t_plus_delta = self.get_coords(time + delta_t)

        # Calculate the velocity
        return (pos_t_plus_delta - pos_t) / delta_t


# Initial conditions for the GPS Constellations
ALMANAC_STATIC = {
    "G02": {
        "ID": 2,
        "Health": 0,
        "eccentricity": 0.01597881317,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9671772836,
        "RateofRightAscen(r/s)": -7.886042771e-09,
        "SQRT(A)(m1/2)": 5153.654785,
        "RightAscenatWeek(rad)": 1.934699938,
        "ArgumentofPerigee(rad)": -1.181754306,
        "MeanAnom(rad)": -0.352432835,
        "Af0(s)": -0.0004253387451,
        "Af1(s/s)": 7.275957614e-12,
        "week": 268,
    },
    "G03": {
        "ID": 3,
        "Health": 0,
        "eccentricity": 0.005547523499,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9848300469,
        "RateofRightAscen(r/s)": -7.680319916e-09,
        "SQRT(A)(m1/2)": 5153.673828,
        "RightAscenatWeek(rad)": 3.069382081,
        "ArgumentofPerigee(rad)": 1.091436319,
        "MeanAnom(rad)": 2.394527906,
        "Af0(s)": 0.0003929138184,
        "Af1(s/s)": 1.818989404e-11,
        "week": 268,
    },
    "G04": {
        "ID": 4,
        "Health": 0,
        "eccentricity": 0.002841472626,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9647804386,
        "RateofRightAscen(r/s)": -8.034620388e-09,
        "SQRT(A)(m1/2)": 5153.684082,
        "RightAscenatWeek(rad)": -2.135634193,
        "ArgumentofPerigee(rad)": -2.946859485,
        "MeanAnom(rad)": -0.9165782382,
        "Af0(s)": 0.000373840332,
        "Af1(s/s)": 7.275957614e-12,
        "week": 268,
    },
    "G05": {
        "ID": 5,
        "Health": 0,
        "eccentricity": 0.005885601044,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9717732339,
        "RateofRightAscen(r/s)": -7.783181344e-09,
        "SQRT(A)(m1/2)": 5153.57373,
        "RightAscenatWeek(rad)": 3.017992601,
        "ArgumentofPerigee(rad)": 1.256599161,
        "MeanAnom(rad)": -0.2147003852,
        "Af0(s)": -0.0001745223999,
        "Af1(s/s)": 0.0,
        "week": 268,
    },
    "G06": {
        "ID": 6,
        "Health": 0,
        "eccentricity": 0.002969264984,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9895758,
        "RateofRightAscen(r/s)": -7.623174679e-09,
        "SQRT(A)(m1/2)": 5153.590332,
        "RightAscenatWeek(rad)": 2.03315409,
        "ArgumentofPerigee(rad)": -0.691689015,
        "MeanAnom(rad)": -2.559430466,
        "Af0(s)": 0.0002422332764,
        "Af1(s/s)": -2.182787284e-11,
        "week": 268,
    },
    "G07": {
        "ID": 7,
        "Health": 63,
        "eccentricity": 0.01857852936,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9499799209,
        "RateofRightAscen(r/s)": -7.851755629e-09,
        "SQRT(A)(m1/2)": 5153.585449,
        "RightAscenatWeek(rad)": -1.12251441,
        "ArgumentofPerigee(rad)": -2.119738617,
        "MeanAnom(rad)": 2.850169635,
        "Af0(s)": -0.0001068115234,
        "Af1(s/s)": 1.091393642e-11,
        "week": 268,
    },
    "G08": {
        "ID": 8,
        "Health": 0,
        "eccentricity": 0.009456157684,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9531377641,
        "RateofRightAscen(r/s)": -8.308917528e-09,
        "SQRT(A)(m1/2)": 5153.580566,
        "RightAscenatWeek(rad)": 0.9368959935,
        "ArgumentofPerigee(rad)": 0.364454136,
        "MeanAnom(rad)": -0.5841489471,
        "Af0(s)": 0.0002088546753,
        "Af1(s/s)": 1.818989404e-11,
        "week": 268,
    },
    "G09": {
        "ID": 9,
        "Health": 0,
        "eccentricity": 0.002718448639,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9584168152,
        "RateofRightAscen(r/s)": -8.126052768e-09,
        "SQRT(A)(m1/2)": 5153.597168,
        "RightAscenatWeek(rad)": -2.195206652,
        "ArgumentofPerigee(rad)": 1.982660431,
        "MeanAnom(rad)": -0.1168038735,
        "Af0(s)": 0.0002183914185,
        "Af1(s/s)": 1.455191523e-11,
        "week": 268,
    },
    "G10": {
        "ID": 10,
        "Health": 0,
        "eccentricity": 0.009332180023,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9844884965,
        "RateofRightAscen(r/s)": -7.703178011e-09,
        "SQRT(A)(m1/2)": 5153.64209,
        "RightAscenatWeek(rad)": 3.066818955,
        "ArgumentofPerigee(rad)": -2.37289301,
        "MeanAnom(rad)": 1.397857595,
        "Af0(s)": -3.433227539e-05,
        "Af1(s/s)": -7.275957614e-12,
        "week": 268,
    },
    "G11": {
        "ID": 11,
        "Health": 0,
        "eccentricity": 0.00131893158,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9661226718,
        "RateofRightAscen(r/s)": -7.851755629e-09,
        "SQRT(A)(m1/2)": 5153.672363,
        "RightAscenatWeek(rad)": 2.066106964,
        "ArgumentofPerigee(rad)": -2.588128565,
        "MeanAnom(rad)": -1.275203921,
        "Af0(s)": -0.0006761550903,
        "Af1(s/s)": -7.275957614e-12,
        "week": 268,
    },
    "G12": {
        "ID": 12,
        "Health": 0,
        "eccentricity": 0.008770465851,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.962773081,
        "RateofRightAscen(r/s)": -7.931758961e-09,
        "SQRT(A)(m1/2)": 5153.596191,
        "RightAscenatWeek(rad)": -0.00704223016,
        "ArgumentofPerigee(rad)": 1.451508723,
        "MeanAnom(rad)": 2.116594256,
        "Af0(s)": -0.0005102157593,
        "Af1(s/s)": -3.637978807e-12,
        "week": 268,
    },
    "G13": {
        "ID": 13,
        "Health": 0,
        "eccentricity": 0.008083343506,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.971245928,
        "RateofRightAscen(r/s)": -7.954617056e-09,
        "SQRT(A)(m1/2)": 5153.600098,
        "RightAscenatWeek(rad)": -2.02919218,
        "ArgumentofPerigee(rad)": 0.92446648,
        "MeanAnom(rad)": -0.853581662,
        "Af0(s)": 0.0006551742554,
        "Af1(s/s)": 3.637978807e-12,
        "week": 268,
    },
    "G14": {
        "ID": 14,
        "Health": 0,
        "eccentricity": 0.00453710556,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9460850478,
        "RateofRightAscen(r/s)": -8.091765626e-09,
        "SQRT(A)(m1/2)": 5153.646973,
        "RightAscenatWeek(rad)": -0.05153890822,
        "ArgumentofPerigee(rad)": -2.90444207,
        "MeanAnom(rad)": 2.095591153,
        "Af0(s)": 0.0004148483276,
        "Af1(s/s)": 1.091393642e-11,
        "week": 268,
    },
    "G15": {
        "ID": 15,
        "Health": 0,
        "eccentricity": 0.01564025879,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9353831349,
        "RateofRightAscen(r/s)": -8.377491813e-09,
        "SQRT(A)(m1/2)": 5153.592773,
        "RightAscenatWeek(rad)": -2.323255848,
        "ArgumentofPerigee(rad)": 1.313138113,
        "MeanAnom(rad)": -1.532751656,
        "Af0(s)": 0.0001659393311,
        "Af1(s/s)": 3.637978807e-12,
        "week": 268,
    },
    "G16": {
        "ID": 16,
        "Health": 0,
        "eccentricity": 0.01390171051,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9622517672,
        "RateofRightAscen(r/s)": -7.954617056e-09,
        "SQRT(A)(m1/2)": 5153.736816,
        "RightAscenatWeek(rad)": 0.01081389044,
        "ArgumentofPerigee(rad)": 0.839942865,
        "MeanAnom(rad)": 0.3180785563,
        "Af0(s)": -0.0002756118774,
        "Af1(s/s)": 1.091393642e-11,
        "week": 268,
    },
    "G17": {
        "ID": 17,
        "Health": 0,
        "eccentricity": 0.01339912415,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9688430909,
        "RateofRightAscen(r/s)": -8.171768958e-09,
        "SQRT(A)(m1/2)": 5153.683105,
        "RightAscenatWeek(rad)": 1.031727165,
        "ArgumentofPerigee(rad)": -1.320440625,
        "MeanAnom(rad)": -0.5810285545,
        "Af0(s)": 0.0006999969482,
        "Af1(s/s)": -3.637978807e-12,
        "week": 268,
    },
    "G18": {
        "ID": 18,
        "Health": 0,
        "eccentricity": 0.004152297974,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9744097633,
        "RateofRightAscen(r/s)": -7.771752296e-09,
        "SQRT(A)(m1/2)": 5153.567383,
        "RightAscenatWeek(rad)": 2.037151204,
        "ArgumentofPerigee(rad)": -3.053585,
        "MeanAnom(rad)": -2.541779201,
        "Af0(s)": -0.0006189346313,
        "Af1(s/s)": -3.637978807e-12,
        "week": 268,
    },
    "G19": {
        "ID": 19,
        "Health": 0,
        "eccentricity": 0.00988483429,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9680101872,
        "RateofRightAscen(r/s)": -8.148910863e-09,
        "SQRT(A)(m1/2)": 5153.655762,
        "RightAscenatWeek(rad)": 1.076056813,
        "ArgumentofPerigee(rad)": 2.550191378,
        "MeanAnom(rad)": 1.40855389,
        "Af0(s)": 0.0004901885986,
        "Af1(s/s)": 3.637978807e-12,
        "week": 268,
    },
    "G20": {
        "ID": 20,
        "Health": 0,
        "eccentricity": 0.003585338593,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9541803917,
        "RateofRightAscen(r/s)": -7.943188009e-09,
        "SQRT(A)(m1/2)": 5153.75,
        "RightAscenatWeek(rad)": 2.889131848,
        "ArgumentofPerigee(rad)": -2.593903089,
        "MeanAnom(rad)": -2.143387238,
        "Af0(s)": 0.0003747940063,
        "Af1(s/s)": 0.0,
        "week": 268,
    },
    "G21": {
        "ID": 21,
        "Health": 0,
        "eccentricity": 0.02516078949,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9614368399,
        "RateofRightAscen(r/s)": -7.908900866e-09,
        "SQRT(A)(m1/2)": 5153.640625,
        "RightAscenatWeek(rad)": 1.927475697,
        "ArgumentofPerigee(rad)": -0.566707031,
        "MeanAnom(rad)": -0.727667905,
        "Af0(s)": 0.0001182556152,
        "Af1(s/s)": -3.637978807e-12,
        "week": 268,
    },
    "G22": {
        "ID": 22,
        "Health": 0,
        "eccentricity": 0.01427221298,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9615746585,
        "RateofRightAscen(r/s)": -7.966046104e-09,
        "SQRT(A)(m1/2)": 5153.65332,
        "RightAscenatWeek(rad)": 0.01410730525,
        "ArgumentofPerigee(rad)": -1.134888122,
        "MeanAnom(rad)": -0.06024619663,
        "Af0(s)": -2.193450928e-05,
        "Af1(s/s)": -3.637978807e-12,
        "week": 268,
    },
    "G23": {
        "ID": 23,
        "Health": 0,
        "eccentricity": 0.004188537598,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9796229012,
        "RateofRightAscen(r/s)": -7.748894201e-09,
        "SQRT(A)(m1/2)": 5153.670898,
        "RightAscenatWeek(rad)": 3.037822748,
        "ArgumentofPerigee(rad)": -2.969010452,
        "MeanAnom(rad)": 2.539609682,
        "Af0(s)": 0.0002403259277,
        "Af1(s/s)": 1.091393642e-11,
        "week": 268,
    },
    "G24": {
        "ID": 24,
        "Health": 0,
        "eccentricity": 0.01531219482,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.933627446,
        "RateofRightAscen(r/s)": -8.068907531e-09,
        "SQRT(A)(m1/2)": 5153.657227,
        "RightAscenatWeek(rad)": -1.219678011,
        "ArgumentofPerigee(rad)": 0.979932843,
        "MeanAnom(rad)": -2.420004496,
        "Af0(s)": -0.0004749298096,
        "Af1(s/s)": -3.637978807e-12,
        "week": 268,
    },
    "G25": {
        "ID": 25,
        "Health": 0,
        "eccentricity": 0.01162719727,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9506929822,
        "RateofRightAscen(r/s)": -8.023191341e-09,
        "SQRT(A)(m1/2)": 5153.733398,
        "RightAscenatWeek(rad)": -0.09362638257,
        "ArgumentofPerigee(rad)": 1.090139026,
        "MeanAnom(rad)": 2.086611223,
        "Af0(s)": 0.0004968643188,
        "Af1(s/s)": 0.0,
        "week": 268,
    },
    "G26": {
        "ID": 26,
        "Health": 0,
        "eccentricity": 0.008870124817,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9314283407,
        "RateofRightAscen(r/s)": -8.206056101e-09,
        "SQRT(A)(m1/2)": 5153.590332,
        "RightAscenatWeek(rad)": -0.1561686818,
        "ArgumentofPerigee(rad)": 0.541729285,
        "MeanAnom(rad)": 1.280126442,
        "Af0(s)": 0.0001440048218,
        "Af1(s/s)": -7.275957614e-12,
        "week": 268,
    },
    "G27": {
        "ID": 27,
        "Health": 0,
        "eccentricity": 0.01254272461,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9621319249,
        "RateofRightAscen(r/s)": -8.206056101e-09,
        "SQRT(A)(m1/2)": 5153.625977,
        "RightAscenatWeek(rad)": 0.9634163342,
        "ArgumentofPerigee(rad)": 0.795262679,
        "MeanAnom(rad)": -0.5175788248,
        "Af0(s)": -2.670288086e-05,
        "Af1(s/s)": 0.0,
        "week": 268,
    },
    "G28": {
        "ID": 28,
        "Health": 0,
        "eccentricity": 0.0003271102905,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9610593368,
        "RateofRightAscen(r/s)": -7.748894201e-09,
        "SQRT(A)(m1/2)": 5153.608887,
        "RightAscenatWeek(rad)": -1.155560161,
        "ArgumentofPerigee(rad)": 1.685558763,
        "MeanAnom(rad)": 1.466470653,
        "Af0(s)": -0.0002698898315,
        "Af1(s/s)": -1.818989404e-11,
        "week": 268,
    },
    "G29": {
        "ID": 29,
        "Health": 0,
        "eccentricity": 0.002945423126,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9712579122,
        "RateofRightAscen(r/s)": -8.137481816e-09,
        "SQRT(A)(m1/2)": 5153.53418,
        "RightAscenatWeek(rad)": 1.046747144,
        "ArgumentofPerigee(rad)": 2.552183381,
        "MeanAnom(rad)": -0.5697218128,
        "Af0(s)": -0.0005960464478,
        "Af1(s/s)": 0.0,
        "week": 268,
    },
    "G30": {
        "ID": 30,
        "Health": 0,
        "eccentricity": 0.007164955139,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.935119482,
        "RateofRightAscen(r/s)": -8.011762293e-09,
        "SQRT(A)(m1/2)": 5153.580566,
        "RightAscenatWeek(rad)": -1.121981112,
        "ArgumentofPerigee(rad)": -2.461502121,
        "MeanAnom(rad)": 2.693733555,
        "Af0(s)": -0.0003814697266,
        "Af1(s/s)": 7.275957614e-12,
        "week": 268,
    },
    "G31": {
        "ID": 31,
        "Health": 0,
        "eccentricity": 0.01047277451,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9539646756,
        "RateofRightAscen(r/s)": -7.806039439e-09,
        "SQRT(A)(m1/2)": 5153.681152,
        "RightAscenatWeek(rad)": -1.100986997,
        "ArgumentofPerigee(rad)": 0.679280473,
        "MeanAnom(rad)": 1.963850067,
        "Af0(s)": -0.0002279281616,
        "Af1(s/s)": 0.0,
        "week": 268,
    },
    "G32": {
        "ID": 32,
        "Health": 0,
        "eccentricity": 0.00778055191,
        "TimeofApplicability(s)": 589824.0,
        "OrbitalInclination(rad)": 0.9618263272,
        "RateofRightAscen(r/s)": -8.068907531e-09,
        "SQRT(A)(m1/2)": 5153.620605,
        "RightAscenatWeek(rad)": -2.180563053,
        "ArgumentofPerigee(rad)": -2.130164518,
        "MeanAnom(rad)": 0.139394512,
        "Af0(s)": -0.0006170272827,
        "Af1(s/s)": 0.0,
        "week": 268,
    },
}


def from_almanac(almanac: tp.Optional[Path] = None) -> dict[str, KepelerianSatellite]:
    """Reads the almanac file and returns a list of KepelerianSatellite objects.

    Args:
        almanac (Path): Path to the almanac file.

    Returns:
        list[KepelerianSatellite]: List of KepelerianSatellite objects.
    """
    # Read the almanac file
    if almanac is None:
        almanac_data = ALMANAC_STATIC
    else:
        almanac_data = IParseYumaAlm()(filepath=almanac)[1].to_dict(orient="index")

    # Create KepelerianSatellite objects
    satellites = {}
    for prn, records in almanac_data.items():
        satellites[prn] = KepelerianSatellite(
            prn=prn,
            sqrtA=records["SQRT(A)(m1/2)"],
            eccentricty=records["eccentricity"],
            time_of_almanac=records["TimeofApplicability(s)"],
            inclination=records["OrbitalInclination(rad)"],
            right_ascension=records["RightAscenatWeek(rad)"],
            rate_of_right_ascension=records["RateofRightAscen(r/s)"],
            argument_of_perigee=records["ArgumentofPerigee(rad)"],
            mean_anomaly=records["MeanAnom(rad)"],
            week=records["week"],
            health=records["Health"],
            af0=records["Af0(s)"],
            af1=records["Af1(s/s)"],
        )

    return satellites
