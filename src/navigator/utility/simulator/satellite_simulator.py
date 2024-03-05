"""GNSS Satellite Trajectory Simulator.

This module defines a Python class, GNSSsatellite, for simulating the trajectory of a GNSS satellite in space. The simulation is based on Keplerian orbital elements, and the class provides a method to calculate the coordinates of the satellite at a given time.

Classes:
    GNSSsatellite: 
        Simulates the trajectory of a GNSS satellite in space.

Usage Example:
    # Initialize a GNSS satellite with Keplerian orbital elements
    satellite = GNSSsatellite(semi_major_axis=26559897, eccentricity=0.01, inclination=55, argument_of_perigee=45, mean_anomaly=30, mean_motion=15.0)

    # Calculate coordinates after 1 hour (3600 seconds)
    coordinates_at_t = satellite.get_coords(3600)

    # Print the coordinates
    print("Coordinates after 1 hour:", coordinates_at_t)

Module Dependencies:
    NumPy: 
        The module utilizes NumPy for numerical calculations.

Attributes:
    semi_major_axis: 
        Semi-major axis of the elliptical orbit (in meters)
    eccentricity: 
        eccentricity of the elliptical orbit
    inclination: 
        Inclination of the orbit in degrees
    argument_of_perigee: 
        Argument of perigee in degrees
    mean_anomaly: 
        Mean anomaly at epoch in degrees
    mean_motion: 
        Mean motion of the satellite in revolutions per day

Methods:
    __init__(self, semi_major_axis, eccentricity, inclination, argument_of_perigee, mean_anomaly, mean_motion): 
        Initializes the GNSSsatellite object with Keplerian orbital elements.
    get_coords(self, t): 
        Returns the coordinates (x, y, z) of the satellite at a given time t.

Note:
    The Keplerian orbital elements (semi_major_axis, eccentricity, inclination, etc.) should be replaced with the actual values of the GNSS satellite you want to simulate.
"""

from ...core.satellite.iephm.sv.tools.coord import ephm_to_coord_gps

__all__ = ['GNSSsatellite', 'GPSConstellation']


class GNSSsatellite:
    """Simulates the trajectory of a GNSS satellite in space.

    Attributes:
    - semi_major_axis: Semi-major axis of the elliptical orbit (in meters)
    - eccentricity: eccentricity of the elliptical orbit
    - inclination: Inclination of the orbit in degrees
    - argument_of_perigee: Argument of perigee in degrees
    - mean_anomaly: Mean anomaly at epoch in degrees
    - mean_motion: Mean motion of the satellite in revolutions per day

    Methods:
    - get_coords(t: float) -> Tuple[float, float, float]: Returns the coordinates (x, y, z) of the satellite at a given time t.
    """

    def __init__(
        self,
        sqrtA: float,
        Eccentricity: float,
        Io: float,
        Omega0: float,
        M0: float,
        DeltaN: float = 0.0,
    ) -> None:
        """Initialize the GNSSsatellite object with Keplerian orbital elements.

        Args:
            sqrtA (float): Square root of semi-major axis of the elliptical orbit (in meters)
            Eccentricity (float): Eccentricity of the elliptical orbit
            Io (float): Inclination of the orbit in radians
            Omega0 (float): Argument of perigee in radians
            M0 (float): Mean anomaly at epoch in radians
            DeltaN (float): Mean motion of the satellite in revolutions per day.
        """
        self.semi_major_axis = sqrtA * 1e3
        self.eccentricity = Eccentricity
        self.inclination = Io
        self.argument_of_perigee = Omega0
        self.mean_anomaly = M0
        self.delta_n = DeltaN

    def get_coords(self, t: float) -> tuple:
        """Calculate the coordinates (x, y, z) of the satellite at a given time t.

        Parameters:
        - t: Time elapsed from the initial time (in seconds)

        Returns:
        - Tuple (x, y, z) representing the coordinates of the satellite
        """
        # Constants
        return ephm_to_coord_gps(
            t=t,
            toe=0,
            sqrt_a=self.semi_major_axis**0.5,
            e=self.eccentricity,
            M_0=self.mean_anomaly,
            w=self.argument_of_perigee,
            i_0=self.inclination,
            omega_0=0,
            delta_n=self.delta_n,
            i_dot=0,
            omega_dot=0,
            c_us=0,
            c_uc=0,
            c_rs=0,
            c_rc=0,
            c_is=0,
            c_ic=0,
        )


class GPSConstellation:
    """Simulates the GPS satellite constillation."""

    INITIAL_CONDITIONS = {
        'G01': {
            'sqrtA': 5153.660242081,
            'Eccentricity': 0.01204742747359,
            'Io': 0.9891789945442,
            'Omega0': -0.5116143834326,
            'DeltaN': 3.801229765049e-09,
            'M0': -0.6681039753151,
        },
        'G02': {
            'sqrtA': 5153.700933456,
            'Eccentricity': 0.02000075648539,
            'Io': 0.9669485791867,
            'Omega0': -0.6066903042292,
            'DeltaN': 4.212675474843e-09,
            'M0': 1.546230090621,
        },
        'G03': {
            'sqrtA': 5153.771314621,
            'Eccentricity': 0.004439325537533,
            'Io': 0.9763548527786,
            'Omega0': 0.5201910595885,
            'DeltaN': 4.28732144129e-09,
            'M0': -2.840534537631,
        },
        'G04': {
            'sqrtA': 5153.736791611,
            'Eccentricity': 0.002154131303541,
            'Io': 0.9615650640617,
            'Omega0': 1.601686421354,
            'DeltaN': 4.508402078757e-09,
            'M0': 0.1808035127235,
        },
        'G05': {
            'sqrtA': 5153.561510086,
            'Eccentricity': 0.005898805800825,
            'Io': 0.9634063293292,
            'Omega0': 0.476081517929,
            'DeltaN': 4.731982820364e-09,
            'M0': 1.856154107492,
        },
        'G06': {
            'sqrtA': 5153.564332962,
            'Eccentricity': 0.002650320879184,
            'Io': 0.9884575916034,
            'Omega0': -0.5199555136856,
            'DeltaN': 3.711583173766e-09,
            'M0': 0.6620175426822,
        },
        'G07': {
            'sqrtA': 5153.622900009,
            'Eccentricity': 0.01645870879292,
            'Io': 0.9507268584584,
            'Omega0': 2.618341132807,
            'DeltaN': 4.544475009911e-09,
            'M0': -2.338985975629,
        },
        'G08': {
            'sqrtA': 5153.619432449,
            'Eccentricity': 0.007800261140801,
            'Io': 0.9610757618517,
            'Omega0': -1.598280000515,
            'DeltaN': 4.694124100539e-09,
            'M0': 0.5471395349692,
        },
        'G09': {
            'sqrtA': 5153.720909119,
            'Eccentricity': 0.002643894287758,
            'Io': 0.9547227533606,
            'Omega0': 1.546489220226,
            'DeltaN': 4.698767151084e-09,
            'M0': 0.9061415612103,
        },
        'G10': {
            'sqrtA': 5153.682167053,
            'Eccentricity': 0.008190797176212,
            'Io': 0.9761397467672,
            'Omega0': 0.5171475185042,
            'DeltaN': 4.434470427779e-09,
            'M0': 2.567805814185,
        },
        'G11': {
            'sqrtA': 5153.672706604,
            'Eccentricity': 0.0008048370946199,
            'Io': 0.9644928649192,
            'Omega0': -0.474351890866,
            'DeltaN': 4.281606917543e-09,
            'M0': 2.05179723007,
        },
        'G12': {
            'sqrtA': 5153.707902908,
            'Eccentricity': 0.008628415758722,
            'Io': 0.9675482161411,
            'Omega0': -2.551343677499,
            'DeltaN': 4.649836541499e-09,
            'M0': 0.06771211565855,
        },
        'G13': {
            'sqrtA': 5153.673830032,
            'Eccentricity': 0.006682727951556,
            'Io': 0.9687821290243,
            'Omega0': 1.703758847343,
            'DeltaN': 4.364110354142e-09,
            'M0': 1.083577753969,
        },
        'G14': {
            'sqrtA': 5153.654602051,
            'Eccentricity': 0.002398941316642,
            'Io': 0.9509331416109,
            'Omega0': -2.586840032956,
            'DeltaN': 5.072354141053e-09,
            'M0': -3.099128015556,
        },
        'G15': {
            'sqrtA': 5153.639408112,
            'Eccentricity': 0.01473562791944,
            'Io': 0.9308411672366,
            'Omega0': 1.433009514964,
            'DeltaN': 5.402725045184e-09,
            'M0': 0.5968220513483,
        },
        'G16': {
            'sqrtA': 5153.741518021,
            'Eccentricity': 0.0130136271473,
            'Io': 0.9673317423013,
            'Omega0': -2.53275059436,
            'DeltaN': 4.798414158924e-09,
            'M0': 1.414200471469,
        },
        'G17': {
            'sqrtA': 5153.662157059,
            'Eccentricity': 0.01386858872138,
            'Io': 0.9764788731217,
            'Omega0': -1.512184715832,
            'DeltaN': 4.190531695323e-09,
            'M0': 1.579598681471,
        },
        'G18': {
            'sqrtA': 5153.717258453,
            'Eccentricity': 0.002958950237371,
            'Io': 0.9730390758061,
            'Omega0': -0.5080921579978,
            'DeltaN': 4.092313318419e-09,
            'M0': -2.573028358582,
        },
        'G19': {
            'sqrtA': 5153.736066818,
            'Eccentricity': 0.009025134611875,
            'Io': 0.9754319840746,
            'Omega0': -1.46748027561,
            'DeltaN': 4.196960534538e-09,
            'M0': -2.597371486582,
        },
        'G20': {
            'sqrtA': 5153.68563652,
            'Eccentricity': 0.004738848540001,
            'Io': 0.9458958201976,
            'Omega0': 0.3571140938869,
            'DeltaN': 5.090569185497e-09,
            'M0': -0.6831181493329,
        },
        'G21': {
            'sqrtA': 5153.682523727,
            'Eccentricity': 0.02450640965253,
            'Io': 0.961270995584,
            'Omega0': -0.6100139253685,
            'DeltaN': 4.374110770699e-09,
            'M0': 1.56039474811,
        },
        'G22': {
            'sqrtA': 5153.758422852,
            'Eccentricity': 0.01353631168604,
            'Io': 0.96154129018,
            'Omega0': 1.649745493653,
            'DeltaN': 4.729839873959e-09,
            'M0': 1.354359463651,
        },
        'G23': {
            'sqrtA': 5153.634134293,
            'Eccentricity': 0.002874290454201,
            'Io': 0.9712571640508,
            'Omega0': 0.4909592850763,
            'DeltaN': 4.563404369823e-09,
            'M0': 2.725174638851,
        },
        'G24': {
            'sqrtA': 5153.672504425,
            'Eccentricity': 0.01344625093043,
            'Io': 0.9342043866425,
            'Omega0': 2.530835287883,
            'DeltaN': 4.985921969377e-09,
            'M0': 0.6806253939796,
        },
        'G25': {
            'sqrtA': 5153.588817596,
            'Eccentricity': 0.0108018493047,
            'Io': 0.9551740050713,
            'Omega0': -2.631476678629,
            'DeltaN': 4.860559604675e-09,
            'M0': 0.9919328822375,
        },
        'G26': {
            'sqrtA': 5153.592493057,
            'Eccentricity': 0.007475657272153,
            'Io': 0.9359940050502,
            'Omega0': -2.683364873169,
            'DeltaN': 5.407368095729e-09,
            'M0': 2.311529408318,
        },
        'G27': {
            'sqrtA': 5153.6606884,
            'Eccentricity': 0.01079550944269,
            'Io': 0.9698891966695,
            'Omega0': -1.576526173146,
            'DeltaN': 4.545189325379e-09,
            'M0': 0.5371347801282,
        },
        'G29': {
            'sqrtA': 5153.619853973,
            'Eccentricity': 0.002124657155946,
            'Io': 0.9786308300051,
            'Omega0': -1.498704054468,
            'DeltaN': 4.231961992489e-09,
            'M0': -1.650469007585,
        },
        'G30': {
            'sqrtA': 5153.685945511,
            'Eccentricity': 0.005939403316006,
            'Io': 0.9358649786014,
            'Omega0': 2.627325868017,
            'DeltaN': 4.928062416438e-09,
            'M0': -2.455302208827,
        },
        'G31': {
            'sqrtA': 5153.62371254,
            'Eccentricity': 0.01073037879542,
            'Io': 0.9548895479654,
            'Omega0': 2.637253431985,
            'DeltaN': 4.415183910132e-09,
            'M0': 2.357683726168,
        },
        'G32': {
            'sqrtA': 5153.671319962,
            'Eccentricity': 0.006219985312782,
            'Io': 0.9584264245248,
            'Omega0': 1.558449345019,
            'DeltaN': 4.78019911448e-09,
            'M0': 2.319507140943,
        },
    }

    def __init__(self) -> None:
        """Initialize the GPSConstillation object."""
        # Initialize GPS satellites
        self.satellites = {
            sat: GNSSsatellite(**self.INITIAL_CONDITIONS[sat])
            for sat in self.INITIAL_CONDITIONS
        }

    def get_coords(self, t: float) -> dict:
        """Calculate the coordinates (x, y, z) of the GPS satellites at a given time t.

        Parameters:
        - t: Time elapsed from the initial time (in seconds)

        Returns:
        - Dictionary of satellite names and their coordinates
        """
        coordinates = {}
        for sat, satellite in self.satellites.items():
            coordinates[sat] = satellite.get_coords(t)
        return coordinates

    def __repr__(self) -> str:
        """Return a string representation of the GPSConstillation object."""
        return f'GPSConstillation(sv={len(self.satellites)})'
