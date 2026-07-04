"""Milestone 6 - Sample names from your trained network."""
import unittest

import torch

from data import load_words, build_vocab
from neural import sample


def vocab():
    return build_vocab(load_words("names.txt"))


class TestSample(unittest.TestCase):
    def test_follows_the_network(self):
        # A rigged network: huge logits force '.' -> 'a' -> 'b' -> '.'.
        # The only realistic sample is "ab".
        _, itos = vocab()
        W = torch.zeros((27, 27))
        W[0, 1] = 50.0   # . -> a
        W[1, 2] = 50.0   # a -> b
        W[2, 0] = 50.0   # b -> .
        for _ in range(3):
            self.assertEqual(sample(W, itos), "ab",
                             "the next character must come from a forward pass "
                             "on the current one - and stop at '.'")

    def test_returns_clean_lowercase_string(self):
        _, itos = vocab()
        g = torch.Generator().manual_seed(2147483647)
        W = torch.randn((27, 27), generator=g)
        gs = torch.Generator().manual_seed(2147483647)
        for _ in range(10):
            name = sample(W, itos, gs)
            self.assertIsInstance(name, str)
            self.assertNotIn(".", name, "strip the start/end token")
            self.assertTrue(all(c.isalpha() and c.islower() for c in name),
                            f"sampled {name!r} - only lowercase letters allowed")

    def test_same_seed_same_names(self):
        _, itos = vocab()
        g = torch.Generator().manual_seed(2147483647)
        W = torch.randn((27, 27), generator=g)
        g1 = torch.Generator().manual_seed(2147483647)
        g2 = torch.Generator().manual_seed(2147483647)
        first = [sample(W, itos, g1) for _ in range(5)]
        second = [sample(W, itos, g2) for _ in range(5)]
        self.assertEqual(first, second,
                         "pass the generator through to torch.multinomial so "
                         "sampling is reproducible")


if __name__ == "__main__":
    unittest.main()
