"""Implementations of the Set-Transformer blocks which are used to build the Set-Transformer model.

The set transformer blocks are taken from:
    - Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks (https://arxiv.org/abs/1810.00825)

Following are the blocks implemented:
    - Multihead Attention Block (MAB)
    - Set Attention Block (SAB)
    - Induced Set Attention Block (ISAB)
    - Pooling Set Attention Block (PSAB)
"""


import torch
import torch.nn as nn

__all__ = ["MAB", "SAB", "ISAB", "PMA"]


class MAB(nn.Module):
    """Multihead Attention Block (MAB) as described in the paper.

    The MAB block is used to compute the attention between the elements of the set. The attention is computed using the
    scaled dot-product attention mechanism. The MAB block is used to compute the self-attention of the set elements.

    Args:
        dim_Q (int): Dimension of the Query vector.
        dim_K (int): Dimension of the Key vector.
        num_heads (int): Number of attention heads.
        ln (bool): If True, apply layer normalization.
        fnn_dim (int): Dimension of the Pointwise Feedforward Network (FNN).

    """

    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        num_heads: int,
        ln: bool = True,
        fnn_dim: int = 64,
    ) -> None:
        """Constructor method for the MAB block.

        Args:
            dim_Q (int): Dimension of the Query vector.
            dim_K (int): Dimension of the Key vector.
            dim_V (int): Dimension of the Value vector.
            num_heads (int): Number of attention heads.
            ln (bool): If True, apply layer normalization.
            fnn_dim (int): Dimension of the Pointwise Feedforward Network (FNN).

        Returns:
            None
        """
        super().__init__()
        # Multihead Attention Layer as described in equation 7 of the paper.
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=dim_Q,
            num_heads=num_heads,
            kdim=dim_K,
            vdim=dim_K,
            dropout=0.0,
            batch_first=True,
        )
        # The Pointwise Feedforward Network (FNN)
        self.rFNN = nn.Sequential(
            nn.Linear(dim_Q, fnn_dim),
            nn.ReLU(),
            nn.Linear(fnn_dim, fnn_dim),
            nn.ReLU(),
            nn.Linear(fnn_dim, dim_Q),
        )
        # Store the ln flag.
        self.ln = ln

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MAB layer.

        Args:
            X (torch.Tensor): Input tensor of shape (B, N, dim_Q).
            Y (torch.Tensor): Input tensor of shape (B, N, dim_K).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim_V).
        """
        # Compute the attention between the elements of the set.
        # The attention is computed using the scaled dot-product attention mechanism.
        H = X + self.multi_head_attention(query=X, key=Y, value=Y)[0]

        # If layer normalization is applied, apply it here.
        H = nn.functional.layer_norm(H, normalized_shape=H.shape[1:]) if self.ln else H

        # Apply the Pointwise Feedforward Network (FNN)
        H = H + self.rFNN(H)

        # If layer normalization is applied, apply it here.
        return (
            nn.functional.layer_norm(H, normalized_shape=H.shape[1:]) if self.ln else H
        )


class SAB(nn.Module):
    """Set Attention Block (SAB) as described in the paper.

    The SAB block is used to compute the attention between the elements of the set. The attention is computed using the
    scaled dot-product attention mechanism. The SAB block is used to compute the self-attention of the set elements.

    Args:
        dim_Q (int): Dimension of the Query vector.
        num_heads (int): Number of attention heads.
        ln (bool): If True, apply layer normalization.
        fnn_dim (int): Dimension of the Pointwise Feedforward Network (FNN).
    """

    def __init__(
        self,
        dim_Q: int,
        num_heads: int,
        ln: bool = True,
        fnn_dim: int = 64,
    ) -> None:
        """Constructor method for the SAB block.

        Args:
            dim_Q (int): Dimension of the Query vector.
            num_heads (int): Number of attention heads to use on MAB.
            ln (bool): If True, apply layer normalization.
            fnn_dim (int): Dimension of the Pointwise Feedforward Network (FNN).

        Returns:
            None
        """
        super().__init__()
        # Multihead Attention Layer as described in equation 8 of the paper.
        self.MAB = MAB(dim_Q, dim_Q, num_heads, ln, fnn_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SAB layer.

        Args:
            X (torch.Tensor): Input tensor of shape (B, N, dim_Q).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim_V).
        """
        # Compute the self-attention of the set elements.
        return self.MAB(X, X)


class ISAB(nn.Module):
    """Induced Set Attention Block (ISAB) as described in the paper.

    The ISAB block is used to compute the attention between the elements of the set. The attention is computed using the
    scaled dot-product attention mechanism. The ISAB block is used to compute the self-attention of the set elements.

    Args:
        dim_Q (int): Dimension of the Query vector.
        dim_I (int): Dimension of the the inducing points.
        num_heads (int): Number of attention heads.
        ln (bool): If True, apply layer normalization.
        fnn_dim (int): Dimension of the Pointwise Feedforward Network (FNN).
    """

    def __init__(
        self,
        dim_Q: int,
        dim_I: int,
        num_heads: int,
        ln: bool = True,
        fnn_dim: int = 64,
    ) -> None:
        """Constructor method for the ISAB block.

        Args:
            dim_Q (int): Dimension of the Query vector.
            dim_I (int): Dimension of the the inducing points.
            num_heads (int): Number of attention heads to use on MAB.
            ln (bool): If True, apply layer normalization.
            fnn_dim (int): Dimension of the Pointwise Feedforward Network (FNN).

        Returns:
            None
        """
        # Call the super constructor.
        super().__init__()
        # Create the learnable seed vector for the Query vector.
        self.I = nn.Parameter(torch.randn(1, dim_I, dim_Q), requires_grad=True)
        # Xaiver initialization of the seed vector.
        nn.init.xavier_uniform_(self.I)

        # Need Two MAB layers as described in equation 9 and 10 of the paper.
        self.MAB_0 = MAB(
            dim_Q=dim_Q, dim_K=dim_I, num_heads=num_heads, ln=ln, fnn_dim=fnn_dim
        )
        self.MAB_1 = MAB(
            dim_Q=dim_I, dim_K=dim_Q, num_heads=num_heads, ln=ln, fnn_dim=fnn_dim
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for the ISAB layer.

        Args:
            X (torch.Tensor): Input tensor of shape (B, N, dim_Q).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim_Q).
        """
        # Compute the self-attention of inducing points.
        H = self.MAB_1(self.I.repeat(X.shape[0], 1, 1), X)
        return self.MAB_0(X, H)


class PMA(nn.Module):
    """Pooling by Multihead Self-Attention Block (PMA) as described in the paper.

    The PSAB block is used pool the elements of the set using the self-attention mechanism.

    Args:
        dim_Q (int): Dimension of the Query vector.
        dim_S (int): Dimension of the the lernable seed vector.
        num_heads (int): Number of attention heads.
        ln (bool): If True, apply layer normalization.
        fnn_dim (int): Dimension of the Pointwise Feedforward Network (FNN).
    """

    def __init__(
        self, dim_Q: int, dim_S: int, num_heads: int, ln: bool = True, fnn_dim: int = 64
    ) -> None:
        """Constructor method for the pooling set attention block.

        Args:
            dim_Q (int): Dimension of the Query vector.
            dim_S (int): Dimension of the the lernable seed vector.
            num_heads (int): Number of attention heads to use on MAB.
            ln (bool): If True, apply layer normalization.
            fnn_dim (int): Dimension of the Pointwise Feedforward Network (FNN).
        """
        super().__init__()
        # Create the learnable seed vector for the Query vector.
        self.S = nn.Parameter(torch.randn(1, dim_S, dim_Q), requires_grad=True)
        # Xaiver initialization of the seed vector.
        nn.init.xavier_uniform_(self.S)

        # Create a rFF layer as described in equation 11 of the paper.
        self.rFF = nn.Sequential(
            nn.Linear(dim_Q, fnn_dim),
            nn.ReLU(),
            nn.Linear(fnn_dim, fnn_dim),
            nn.ReLU(),
            nn.Linear(fnn_dim, dim_Q),
        )
        # Create a MAB layer as described in equation 11
        self.MAB = MAB(
            dim_Q=dim_Q, dim_K=dim_Q, num_heads=num_heads, ln=ln, fnn_dim=fnn_dim
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for the PSAB layer.

        Args:
            X (torch.Tensor): Input tensor of shape (B, N, dim_Q).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim_Q).
        """
        # Compute the self-attention of inducing points.
        return self.MAB(self.S.repeat(X.shape[0], 1, 1), self.rFF(X))


# Path: src/navigator/neural/set_transformer/blocks/set_attention_blocks.py
