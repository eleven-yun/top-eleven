import os
import sys
import torch
import torch.nn as nn

cwd = os.getcwd()  # current working directory
cfp = os.path.dirname(os.path.abspath(__file__))  # current file path
os.chdir(cfp)
sys.path.append("..")
from encoder.top_encoder import TopEncoder
from decoder.top_decoder import TopDecoder
from embedding.match_token_embedding import MatchTokenEmbedding
os.chdir(cwd)

class TopFormer(nn.Module):
    """The model is a standard Transformer which follows the general encoder-decoder framework."""
    def __init__(self, num_layers: int, d_model: int, nhead: int, num_classes: int):
        """Initialize the transformer with a pair of encoder and decoder blocks, followed by an output layer.

        Parameters
        ----------
        num_layers : int
            The number of identical encoder/decoder layers.
        d_model : int
            The dimension of input/output feature representation.
        nhead : int
            The number of attention heads.
        num_classes : int
            The number of classes.

        """
        super(TopFormer, self).__init__()
        self.encoder = TopEncoder(num_layers, d_model, nhead)
        self.decoder = TopDecoder(num_layers, d_model, nhead)
        self.match_token_embedding = MatchTokenEmbedding(d_model=d_model)
        self.decoder_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.decoder_query, mean=0.0, std=0.02)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x, u=None):
        """Receives input sequences to encoder and decoder and output the classification probabilities.

        Parameters
        ----------
        x : Tensor
            The input sequence to encoder block.
        x : Tensor
            Either token values [batch_size, seq_len] or embedded sequence [seq_len, batch_size, d_model].

        u : Tensor or None
            Optional decoder input sequence [tgt_len, batch_size, d_model].
        Tensor
            The probability output.

        """
        if u is None:
            # Token values path: build batch-first embeddings then convert to seq-first for Transformer.
            token_embeddings = self.match_token_embedding(x)
            x = token_embeddings.permute(1, 0, 2)
            batch_size = x.size(1)
            u = self.decoder_query.expand(-1, batch_size, -1)
        elif x.dim() == 3:
            # Layout compatibility: accept either seq-first or batch-first embeddings.
            if x.size(1) == u.size(1):
                pass
            elif x.size(0) == u.size(1):
                x = x.permute(1, 0, 2)
            else:
                raise ValueError(
                    f"Cannot infer x layout from shapes x={tuple(x.shape)} and u={tuple(u.shape)}"
                )

        z = self.encoder.forward(x)
        y = self.decoder.forward(u, z)
        p = self.output_layer(y)
        return p
