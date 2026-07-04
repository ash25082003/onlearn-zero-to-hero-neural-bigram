"""Milestone 3 - The forward pass: softmax written by hand."""
import unittest

import torch

from neural import forward


class TestForward(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(2147483647)
        self.W = torch.randn((27, 27), generator=g, requires_grad=True)
        self.xs = torch.tensor([0, 5, 13, 13, 1])

    def test_shape(self):
        probs = forward(self.xs, self.W)
        self.assertEqual(tuple(probs.shape), (5, 27),
                         "one row of 27 probabilities per input character")

    def test_rows_are_distributions(self):
        probs = forward(self.xs, self.W)
        self.assertTrue(bool((probs > 0).all().item()),
                        "exp() keeps everything positive")
        self.assertTrue(torch.allclose(probs.sum(1), torch.ones(5), atol=1e-4),
                        "each ROW must sum to 1 (keepdim strikes again)")

    def test_zero_weights_give_uniform(self):
        W0 = torch.zeros((27, 27), requires_grad=True)
        probs = forward(self.xs, W0)
        self.assertTrue(torch.allclose(probs, torch.full((5, 27), 1 / 27), atol=1e-6),
                        "all-zero logits mean the model has no opinion: "
                        "every character equally likely")

    def test_differentiable(self):
        probs = forward(self.xs, self.W)
        self.assertIsNotNone(probs.grad_fn,
                             "the forward pass must stay on PyTorch's autograd "
                             "tape - no .item(), .detach() or numpy detours")


if __name__ == "__main__":
    unittest.main()
