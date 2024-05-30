"""Implementation of the xLSTM architecture as described in the xLSTM paper.

This module provides an implementation of the mLSTM model, a variant of LSTM cells proposed in the xLSTM paper.

Classes:
    - mLSTM: A variant of LSTM cells proposed in the xLSTM paper.
    - sLSTM: A variant of mLSTM cells proposed in the xLSTM paper.
    - mLSTMCell: A single mLSTM cell.
    - sLSTMCell: A single sLSTM cell.

See Also:
    'navigator.neural.architectures.set_transformer' for the implementation of the Set Transformer model.
    'navigator.neural.architectures.kalman_nets' for the implementation of the Kalman Networks model.

References:
    - xLSTM: Extended Long Short-Term Memory
      https://arxiv.org/abs/2405.04517

Examples:
    >>> model = mLSTM(input_size=128, hidden_size=256, num_layers=2)
    >>> input_tensor = torch.randn(10, 32, 128)  # seq_len=10, batch_size=32, input_size=128
    >>> output, (H, C, N, M) = model(input_tensor)
"""

from .mLSTM import mLSTM, mLSTMCell
from .sLSTM import sLSTM, sLSTMCell
