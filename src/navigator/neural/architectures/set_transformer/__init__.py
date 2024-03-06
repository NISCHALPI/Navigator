"""Implementations of the Set-Transformer blocks which are used to build the Set-Transformer model.

The set transformer blocks are taken from:
    - Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks (https://arxiv.org/abs/1810.00825)

Following are the blocks implemented:
    - Multihead Attention Block (MAB)
    - Set Attention Block (SAB)
    - Induced Set Attention Block (ISAB)
    - Pooling Set Attention Block (PSAB)
"""

from .set_attention_blocks import ISAB, MAB, PMA, SAB
