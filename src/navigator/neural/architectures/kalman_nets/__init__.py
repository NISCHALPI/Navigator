"""This module contains the KalmanNet filter implementations.

KalmanNets are deep learning-accelerated Kalman filters that leverage neural networks to approximate the state of the system.
This package encompasses the KalmanNet and its variants, including the GRUKalmanNet, LSTMKalmanNet, TransformerKalmanNet, and more.

Design Pattern:
    The KalmanNet and its variants are structured as classes inheriting from the KalmanNetBase class, fostering easy extensibility
    and modularity. Additionally, the KalmanNetBase class closely mirrors the structure of the `filterpy.kalman.KalmanFilter`
    class, enabling it to be seamlessly integrated as a drop-in replacement for traditional Kalman filters in existing codebases.

Usage:
    The KalmanNet and its variants can be subclassed to craft custom KalmanNet filters tailored to specific applications. It is
    imperative to train the KalmanNet on a dataset before utilizing it for filtering and interpolation purposes.

Backend DL Framework:
    - PyTorch 
        PyTorch is the default deep learning framework used to implement the KalmanNet and its variants. The PyTorch
        implementation is chosen for its flexibility, ease of use, and extensive support for neural network architectures.
    
    - PyTorch Lightning
        PyTorch Lightning is a lightweight PyTorch wrapper that provides a high-level interface for training and deploying
        PyTorch models. The KalmanNet and its variants leverage PyTorch Lightning to streamline the training and evaluation
        process.

Example:
    ```python
    from navigator.filters import KalmanNet, GRUKalmanNet, LSTMKalmanNet, TransformerKalmanNet

    # Instantiate KalmanNet and its variants
    kalman_net_instance = KalmanNet()
    gru_kalman_net_instance = GRUKalmanNet()
    lstm_kalman_net_instance = LSTMKalmanNet()
    transformer_kalman_net_instance = TransformerKalmanNet()
    ```

Author:
    Nischal Bhattarai (nischalbhattaraipi@gmail.com)
"""

from .gru_knets.gru_extended_kalman_net import GRUExtendedKalmanBlock
from .gru_knets.gru_kalman_net import GRUKalmanBlock
from .kalman_net_base import AbstractKalmanNet
