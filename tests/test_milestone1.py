"""Milestone 1 - The training set."""
import unittest

import torch

from data import load_words, build_vocab
from neural import build_dataset


class TestBuildDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = load_words("names.txt")
        cls.stoi, cls.itos = build_vocab(cls.words)
        cls.xs, cls.ys = build_dataset(cls.words, cls.stoi)

    def test_returns_tensors(self):
        self.assertIsInstance(self.xs, torch.Tensor)
        self.assertIsInstance(self.ys, torch.Tensor)
        self.assertFalse(self.xs.dtype.is_floating_point,
                         "xs holds character indices - keep them integers")

    def test_one_example_per_bigram(self):
        self.assertEqual(self.xs.nelement(), 228146,
                         "every name of length L contributes L+1 examples")
        self.assertEqual(self.ys.nelement(), 228146)

    def test_first_word_emma(self):
        # 'emma' wrapped in dots: .e em mm ma a.
        self.assertEqual(self.xs[:5].tolist(), [0, 5, 13, 13, 1])
        self.assertEqual(self.ys[:5].tolist(), [5, 13, 13, 1, 0],
                         "ys[k] is the character that FOLLOWS xs[k]")

    def test_targets_shifted_by_one(self):
        # inside a single name, today's target is tomorrow's input
        self.assertEqual(self.xs[1:4].tolist(), self.ys[0:3].tolist())

    def test_all_indices_in_vocab(self):
        for t in (self.xs, self.ys):
            self.assertGreaterEqual(int(t.min().item()), 0)
            self.assertLessEqual(int(t.max().item()), 26)


if __name__ == "__main__":
    unittest.main()
