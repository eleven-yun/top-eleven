import unittest
import os
import sys

import torch

cwd = os.getcwd()
cfp = os.path.dirname(os.path.abspath(__file__))
os.chdir(cfp)
sys.path.append(os.path.abspath(".."))
from embedding.match_token_embedding import MatchTokenEmbedding
from embedding.token_schema import TOKEN_COUNT
os.chdir(cwd)


class TestMatchTokenEmbedding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.d_model = 64

    def test_forward_shape(self):
        module = MatchTokenEmbedding(d_model=self.d_model)
        token_values = torch.randn(self.batch_size, TOKEN_COUNT)

        with torch.no_grad():
            y = module(token_values)

        self.assertEqual(y.shape, (self.batch_size, TOKEN_COUNT, self.d_model))


if __name__ == "__main__":
    unittest.main()
