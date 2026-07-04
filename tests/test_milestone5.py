"""Milestone 5 - Train it (gradient descent on the full dataset)."""
import unittest

import torch

from data import load_words, build_vocab
from neural import build_dataset, init_weights, forward, loss_fn
from train import train


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        words = load_words("names.txt")
        stoi, _ = build_vocab(words)
        cls.xs, cls.ys = build_dataset(words, stoi)

    def test_loss_falls_toward_the_counting_model(self):
        g = torch.Generator().manual_seed(42)
        W = init_weights(g)
        before = float(loss_fn(forward(self.xs, W), self.ys, W))
        final = train(self.xs, self.ys, W, steps=25, lr=50)
        self.assertLess(float(final), before - 0.5,
                        "25 steps at lr=50 should drop the loss by a lot "
                        "(from ~3.8) - are you actually updating W.data?")
        self.assertLess(float(final), 2.60,
                        "after 25 steps the loss should already be near 2.56, "
                        "closing in on the counting model's 2.454")

    def test_weights_actually_move(self):
        g = torch.Generator().manual_seed(1337)
        W = init_weights(g)
        w_before = W.detach().clone()
        train(self.xs, self.ys, W, steps=2, lr=50)
        self.assertFalse(torch.equal(w_before, W.detach()),
                         "training must modify W in place (W.data += ...)")
        self.assertIsNotNone(W.grad, "each step needs loss.backward()")


if __name__ == "__main__":
    unittest.main()
