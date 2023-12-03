# Navigator: GNSS Processing Library

Navigator is a educational Python package designed for GNSS (Global Navigation Satellite System) processing, specifically focusing on GPS. It offers functionalities to estimate user positions using GPS observations and navigation data, RINEX data processing pipelines, and data collection scripts. 
Currently, no filtering techniques are implemented, so the estimated positions are not precise. However, the package is under development and will be updated with more functionalities in the future.

## Features

- Triangulation using GPS observations and navigation data
- RINEX data processing, parsing, and plotting
- Data collection scripts form CCDIS and IGS
- Intutuitive and easy-to-use API

## Installation

You can install Navigator using clone this repository and install it using pip:

```bash
git clone $repo_url
cd navigator
pip install .
```

## Usage
Navigator is mostly used as a library, but it also offers CLI tools for data processing and traingulation. See the `docs` directory for more information about the API.

### CLI tools
#### rinex3-download-nasa-mounted
Helps download RINEX data from NASA's CDDIS server. The script will download all the RINEX files from the given start date to the end date. The downloaded files will be stored in the given directory. This requires `curlftpfs` to be installed.
```bash
Usage: rinex3-download-nasa-mounted [OPTIONS] COMMAND [ARGS]...

  Download RINEX files from CCIDS using curlftpfs(Required).

Options:
  -e, --email TEXT  Email to login to CCIDS. Default: hades@PNTFLABCOMP5
  --version         Show the version and exit.
  --help            Show this message and exit.

Commands:
  daily   Download RINEX files for the given year and day.
  yearly  Download RINEX files for the given year.
```
#### rinex3-download-nasa
This is upgraded version of `rinex3-download-nasa-mounted`. This script will download all the RINEX files from the given start date to the end date. The downloaded files will be stored in the given directory. Does not require `curlftpfs` to be installed.
```bash
Usage: rinex3-dir-ops [OPTIONS] COMMAND [ARGS]...

  Epochify RINEX Directory .i.e convert RINEX files direcotry to Epoch
  Directory.

Options:
  -v, --verbose  Enable verbose logging
  --help         Show this message and exit.

Commands:
  epochify    Epochify data contained in RINEX directory.
  standerize  Standerize RINEX V3 files in RINEX directory.
```
#### rinex3-dir-ops
This script will perform operations on the given directory. User can standerize a RINEX data directory, or epochify a standered RINEX data directory that can work
seamlessly with the navigator library. 
```bash
Usage: rinex3-dir-ops [OPTIONS] COMMAND [ARGS]...

  Epochify RINEX Directory .i.e convert RINEX files direcotry to Epoch
  Directory.

Options:
  -v, --verbose  Enable verbose logging
  --help         Show this message and exit.

Commands:
  epochify    Epochify data contained in RINEX directory.
  standerize  Standerize RINEX V3 files in RINEX directory.
```

#### triangulate-gpsv3
This script will triangulate the given RINEX files and print the estimated positions.
```bash
Usage: triangulate-gpsv3 [OPTIONS] COMMAND [ARGS]...

  Triangulate the data from the RINEX files.

Options:
  -v, --verbose  Enable verbose logging
  --help         Show this message and exit.

Commands:
  all-epochs  Triangulate all the epochs in the RINEX files and returns...
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Also, please make sure to update tests as appropriate. (Only PNTF Lab members under navigation department can contribute to this repository.)


## License
This software is strictly licensed for use within the PNTF Lab at the University of Alabama. Usage is solely granted to active members of the PNTF Lab for academic and research purposes. No distribution rights are granted for any member. For more details, refer to the [LICENSE](LICENSE) file.
