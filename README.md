# Navigator: GNSS Processing Library
The Navigator is a python based GNSS library and toolkit tailored for GNSS PVT solutions. It provides a uniform object-oriented API to do baisc GNSS data processing, including RINEX parsing, satellite position calculation, and user PVT calculation. The library is well documented and easy to use. It also provides a set of CLI tools for RINEX data collection from public FTP servers.

## Features
- RINEX data collection, parsing, and processing using simple API
- Satellite Tracking and Position Estimation using Broadcast Ephemeris and SP3 files
- Single Point Positioning (SPP) using WLS and UKF
- Easy to use CLI tools for bulk processing of RINEX data

## Installation
To install the library, user need to clone the repository from **Pntf Lab Server**(*10.116.24.69*) which is only accessible to authorized lab members. To use the **Lab Git Server**, user need to have access to the git user account. If you don't have access to the git user account, please contact the lab administrator.

**Note: The server is only accessible from the UA network. If you are not on the UA network, you need to connect to the UA VPN first.**

To install the library, first create a python virtual environment and activate it. 
**Note: This assumes that you have python3.10 installed on your system.**
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

Now, after activating the virtual environment, you can install the library from the git server using a single command:
```bash
pip install git+ssh://git@10.116.24.69:/home/git/Navigator.git
```
**Note: A password prompt will appear if you are not using an SSH key.**

## Documentation
The illustration of basic usage of library is provided in the `/docs` directory of the repository. To generate the API documentation, activate the virtual environment and run the following command:
```bash
pdoc -o $DOC_DIR -d google navigator
```
where `$DOC_DIR` is the directory where the documentation will be generated. To view the documentation, open the `index.html` file in the `$DOC_DIR/` directory in a web browser.

## Usage
The introduction usage of the library is documented in the *docs* directory. It provides basic usage of the library and its modules. Curretly available introductory notebooks are:
1. [Intro to Parsing](./docs/intro/intro_parsing_and_interpolation.ipynb)
2. [Intro to Traingulation](./docs/intro/intro_triangulation.ipynb)
3. [Intro to Epoch Directory](./docs/intro/epoch_directory_tutorial.ipynb)
4. [Intro to SP3 Orbit](./docs/intro/intro_sp3_orbit.ipynb)
5. [Intro to Unscented Kalman Filter](./docs//intro/unscented_kalman_filter_gps.ipynb)

Other notebooks will be added in the future to provide more detailed usage of the library.

### Data Collection
The data collection is done by using publicly available FTP servers. The primary source of the data is the [CDDIS](https://cddis.nasa.gov/Data_and_Derived_Products/CDDIS_Archive_Access.html) server. The data is collected using the download tools provided by the library.

There are two way of downloading the files from the FTP server.

- Using API provided by download module
- Using CLI tools provided by navigator library (Recommended)

 Two CLI tools are available for downloading the data. These tools are:
- rinex3-download-nasa (For downloading RINEXv3 files)
- rinex3-dir-ops (For standerdizing the directory structure of the downloaded files)

The command line tools are accessible only after activating the virtual environment where the navigator library is installed. To activate the virtual environment, run the following command:
```bash
source .navigator/bin/activate
```

For convenience, the API tools are demonstrated in the [Data Collection](./docs/reports/report-jan-2024/summary_jan24.ipynb)

### Triangulation and User Position Calculation
These are throughly demonstrated in the [Triangulation](./docs/reports/report-jan-2024/summary_jan24.ipynb) notebook.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Also, please make sure to update tests as appropriate. (Only PNTF Lab members under navigation department can contribute to this repository.)

## License
This software is strictly licensed for use within the PNTF Lab at the University of Alabama. Usage is solely granted to active members of the PNTF Lab for academic and research purposes. No distribution rights are granted for any member. For more details, refer to the [LICENSE](LICENSE) file.
