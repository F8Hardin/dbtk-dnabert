import copy
import torch
import torch.nn as nn
from mamba_ssm import Mamba3


class BiMamba3Block(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        mamba_kwargs = dict(
            d_model=embed_dim,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank
        )

        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(embed_dim)
        self.forward_mamba = Mamba3(**mamba_kwargs)
        self.backward_mamba = Mamba3(**mamba_kwargs) if bidirectional else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, **kwargs):
        # kwargs absorbs the transformer-style arguments (src_mask, etc.) so
        # this block is substitutable wherever an encoder block is expected.
        x = self.norm(src)
        output = self.forward_mamba(x)
        if self.backward_mamba is not None:
            output = output + self.backward_mamba(x.flip(-2)).flip(-2)
        return src + self.dropout(output)

    @property
    def embed_dim(self) -> int:
        return self.forward_mamba.d_model


class Mamba3Encoder(nn.Module):

    def __init__(
        self,
        encoder_layer: BiMamba3Block,
        num_layers: int,
        final_layer_norm: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.embed_dim) if final_layer_norm else nn.Identity()

    def forward(self, src: torch.Tensor, **kwargs):
        output = src
        for layer in self.layers:
            output = layer(output)
        return self.norm(output)

    @property
    def embed_dim(self) -> int:
        return self.layers[0].embed_dim

    def __len__(self) -> int:
        return len(self.layers)
