import unittest
import torch

from top_former import TopFormer

class TestTopFormer(unittest.TestCase):
    def setUp(self):
        """Construct a transformer according to specified parameters."""
        self.num_layers = 6
        self.d_model = 512
        self.nhead = 8
        self.num_classes = 3
        self.batch_size = 32
        self.enc_seq_len = 50
        self.dec_seq_len = 10
        self.top_former = TopFormer(self.num_layers, self.d_model, self.nhead, self.num_classes)

    def test_output_shape(self):
        """Check if the output has the expected dimension."""
        x = torch.rand(self.enc_seq_len, self.batch_size, self.d_model)
        u = torch.rand(self.dec_seq_len, self.batch_size, self.d_model)
        y = self.top_former.forward(x, u)
        self.assertEqual(y.size(dim=0), self.dec_seq_len)
        self.assertEqual(y.size(dim=1), self.batch_size)
        self.assertEqual(y.size(dim=2), self.num_classes)

    def test_probability_sum(self):
        """Check if softmax-normalized output probabilities add up to one."""
        x = torch.rand(self.enc_seq_len, self.batch_size, self.d_model)
        u = torch.rand(self.dec_seq_len, self.batch_size, self.d_model)
        logits = self.top_former.forward(x, u)
        p = torch.softmax(logits, dim=2)
        p_sum = torch.sum(p, dim=2)
        eps = 1e-6
        self.assertTrue(torch.all(torch.abs(p_sum - 1.0) < eps))


if __name__ == '__main__':
    unittest.main()
