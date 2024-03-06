"""Neural Network Architectures Module.

This module contains various neural network architectures utilized in the navigator module.

Available Architectures:
    - KalmanNet: A neural network architecture for Kalman Filter. This architecture is designed to enhance Kalman Filter performance using deep learning techniques.
    - SetTransformer: A neural network architecture that acts on sets of data. This architecture is based on the Set Transformer model proposed by Lee et al. (https://arxiv.org/abs/1810.00825), which is specifically designed for set-structured data.

Backend:
    - `torch.nn`: PyTorch's neural network module.

Author:
    Nischal Bhattarai (nischalbhattaraipi@gmail.com)

See Also:
    - `navigator.neural.arch.kalman_nets`: Kalman Filter Neural Network Architectures.
    - `navigator.neural.arch.set_transformer`: Set Transformer Neural Network Architectures.

"""

__author__ = "Nischal Bhattarai"
__email__ = "nischalbhattaraipi@gmail.com"

from .kalman_nets.gru_knets.gru_extended_kalman_net import GRUExtendedKalmanBlock
from .kalman_nets.gru_knets.gru_kalman_net import GRUKalmanBlock
from .kalman_nets.kalman_net_base import AbstractKalmanNet
from .set_transformer.set_attention_blocks import ISAB, MAB, PMA, SAB
