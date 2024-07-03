"""Navigation Neural Network Architectures.

Contains all the neural network-related architecture for GPS/GNSS navigation.

Modules:
  - architectures: Contains the neural network architectures from papers and research.
  - interfaces: Contains the interfaces with the navigator module which implement the neural network triangulation.

Backend:
  - **torch:** PyTorch is utilized as the backend for implementing these neural network architectures, harnessing its powerful capabilities for efficient computation and training.
"""

# Path: src/navigator/neural/__init__.py
from .interface.dynamics_model import (
    BiasedObservationModel,
    DiagonalSymmetricPositiveDefiniteMatrix,
    ObservationModel,
    SymmetricPositiveDefiniteMatrix,
    TransitionModel,
    discretized_process_noise_matrix,
)
from .interface.triangulation_filter import ParametricExtendedInterface
