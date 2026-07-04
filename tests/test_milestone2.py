"""Milestone 2 - One-hot encoding & the weight matrix."""
import unittest

import torch

from neural import encode, init_weights


class TestEncode(unittest.TestCase):
    def test_shape_and_dtype(self):
        xenc = encode(torch.tensor([0, 5, 13]))
        self.assertEqual(tuple(xenc.shape), (3, 27))
        self.assertTrue(xenc.dtype.is_floating_point,
                        "one_hot gives integers - the network needs floats")

    def test_exactly_one_hot(self):
        xs = torch.tensor([0, 5, 13, 26])
        xenc = encode(xs)
        self.assertTrue(torch.allclose(xenc.sum(1), torch.ones(4)),
                        "each row must contain a single 1")
        for row, ix in zip(xenc, xs):
            self.assertEqual(row[ix].item(), 1.0,
                             "the 1 must sit at the character's index")


class TestInitWeights(unittest.TestCase):
    def test_shape(self):
        W = init_weights()
        self.assertEqual(tuple(W.shape), (27, 27),
                         "27 inputs, 27 outputs - one column per neuron")

    def test_requires_grad(self):
        self.assertTrue(init_weights().requires_grad,
                        "without requires_grad=True, PyTorch won't "
                        "backpropagate into W")

    def test_seeded_and_random(self):
        g1 = torch.Generator().manual_seed(2147483647)
        g2 = torch.Generator().manual_seed(2147483647)
        self.assertTrue(torch.equal(init_weights(g1), init_weights(g2)),
                        "pass the generator through to torch.randn")
        self.assertGreater(init_weights().std().item(), 0.5,
                           "weights should start as random normals, not zeros")


if __name__ == "__main__":
    unittest.main()
