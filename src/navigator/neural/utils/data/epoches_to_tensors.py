"""Makes a dataset from RINEX files for GNSS data."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from ....core import IterativeTriangulationInterface
from ....core.triangulate.itriangulate.preprocessor.gps_preprocessor import (
    GPSPreprocessor,
)
from ....epoch import Epoch, EpochCollection, from_rinex_files

__all__ = [
    "rinex_to_contiguous_epoches",
    "epoches_to_preprocessed_tensor",
]


def rinex_to_contiguous_epoches(
    obs_path: Path,
    nav_path: Path | None = None,
    column_mapper: dict[str, str] | None = None,
    **kwargs,
) -> list[EpochCollection]:
    """Converts RINEX files to contiguous EpochCollections.

    Args:
        obs_path: Path to the RINEX observation file.
        nav_path: Path to the RINEX navigation file.
        column_mapper: A dictionary mapping the column names to the desired column names.
        **kwargs: Additional keyword arguments to pass to the from_rinex_files function.

    Returns:
        A list of EpochCollections.
    """
    epoches = list(
        from_rinex_files(
            observation_file=obs_path,
            navigation_file=nav_path,
            column_mapper=column_mapper,
            **kwargs,
        )
    )

    collection = EpochCollection(epoches)
    return collection.track()


def epoches_to_preprocessed_tensor(
    ctgs_epoches: list[Epoch],
    code_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Preprocesses the Epoches into a time series of tensors.

    Args:
        ctgs_epoches: A list of contiguous epoches i.e epoches that are in sequence having same satellites.
        code_only: A boolean indicating whether to use code only.

    Returns:
        dict: A dictionary containing the preprocessed tensors.
    """
    # Preprocesssor
    iterativeInterface = IterativeTriangulationInterface(code_only=code_only)

    # Calculate the sv_map of the initial epoch to track the same satellites in the subsequent epoches
    sv_map = ctgs_epoches[0].obs_data.index

    z, sv_pos = [], []

    fixes = []

    with tqdm.tqdm(ctgs_epoches) as pbar:
        for epoch in pbar:
            epoch.approximate_coords = fixes[-1] if len(fixes) > 0 else None
            z_, sv_pos_ = iterativeInterface._preprocess(
                epoch=epoch,
                sv_filter=sv_map,
                code_only=code_only,
                computational_format=True,
            )
            z.append(z_)
            sv_pos.append(sv_pos_)

            pbar.set_description(f"Processing {epoch.timestamp}")

    code_data_ts = np.vstack(z)
    satellite_data_ts = np.stack(sv_pos, axis=0)

    # Convert to tensors.
    code_data_ts = torch.from_numpy(code_data_ts).to(dtype=torch.float32)
    satellite_data_ts = torch.from_numpy(satellite_data_ts).to(dtype=torch.float32)

    # Set up the initial prior from crude WLS solution.
    soln = GPSPreprocessor().bootstrap(epoch=ctgs_epoches[0])

    x0 = torch.zeros(8, dtype=code_data_ts.dtype)
    x0[0] = soln["x"]
    x0[2] = soln["y"]
    x0[4] = soln["z"]
    x0[6] = soln["cdt"]

    P0 = torch.eye(8, dtype=code_data_ts.dtype)
    P0[:, [0, 2, 4, 6]] *= 100
    P0[:, [1, 3, 5, 7]] *= 10

    return {
        "z": code_data_ts,
        "sv_pos": satellite_data_ts,
        "x0": x0,
        "P0": P0,
    }
