"""Milestone 4 - The loss: negative log-likelihood + regularization."""
import unittest

import torch

from neural import forward, loss_fn


class TestLossFn(unittest.TestCase):
    def test_tiny_example(self):
        # Two examples; the model gives the correct answers probability 0.5 and 0.8.
        probs = torch.tensor([[0.5, 0.25, 0.25],
                              [0.1, 0.8, 0.1]])
        ys = torch.tensor([0, 1])
        W = torch.zeros((2, 2))  # reg term contributes exactly 0
        expected = -(torch.log(torch.tensor(0.5)) + torch.log(torch.tensor(0.8))).item() / 2
        got = loss_fn(probs, ys, W, reg=0.01)
        self.assertAlmostEqual(float(got), expected, places=4,
                               msg="average the -log of the probability given "
                                   "to each CORRECT next character")

    def test_uniform_probs_score_log27(self):
        probs = torch.full((100, 27), 1 / 27)
        ys = torch.randint(0, 27, (100,))
        W = torch.zeros((27, 27))
        self.assertAlmostEqual(float(loss_fn(probs, ys, W)), 3.2958, places=3,
                               msg="a clueless model scores log(27), same as "
                                   "Project 1")

    def test_regularization_term(self):
        probs = torch.full((10, 27), 1 / 27)
        ys = torch.zeros(10, dtype=torch.long)
        W = torch.ones((27, 27))
        base = float(loss_fn(probs, ys, W, reg=0.0))
        with_reg = float(loss_fn(probs, ys, W, reg=0.1))
        self.assertAlmostEqual(with_reg - base, 0.1, places=4,
                               msg="regularization adds reg * (W**2).mean() - "
                                   "for all-ones W that's exactly reg")

    def test_backward_reaches_the_weights(self):
        g = torch.Generator().manual_seed(42)
        W = torch.randn((27, 27), generator=g, requires_grad=True)
        xs = torch.tensor([0, 5, 13, 13, 1])
        ys = torch.tensor([5, 13, 13, 1, 0])
        loss = loss_fn(forward(xs, W), ys, W)
        loss.backward()
        self.assertIsNotNone(W.grad, "loss.backward() must fill W.grad")
        self.assertGreater(W.grad.abs().sum().item(), 0.0,
                           "the gradient shouldn't be all zeros")


if __name__ == "__main__":
    unittest.main()
