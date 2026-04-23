import os
import sys
import torch.nn as nn

cwd = os.getcwd()  # current working directory
cfp = os.path.dirname(os.path.abspath(__file__))  # current file path
os.chdir(cfp)
sys.path.append("..")
from encoder.top_encoder import TopEncoder
from decoder.top_decoder import TopDecoder
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
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x, u):
        """Receives input sequences to encoder and decoder and output the classification probabilities.

        Parameters
        ----------
        x : Tensor
            The input sequence to encoder block.
        u : Tensor
            The input sequence to decoder block.

        Returns
        -------
        Tensor
            The probability output.

        """
        z = self.encoder.forward(x)
        y = self.decoder.forward(u, z)
        p = self.output_layer(y)
        return p
