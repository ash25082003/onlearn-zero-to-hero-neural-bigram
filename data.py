"""
data.py - dataset loading and the character vocabulary.

This file is PROVIDED and complete - you built these two functions yourself in
Project 1 (the bigram language model), so here they're simply given. You don't
edit this file; you import from it.
"""


def load_words(path="names.txt"):
    """Return the 32,033 names, one string per entry."""
    with open(path) as f:
        return f.read().splitlines()


def build_vocab(words):
    """Return (stoi, itos): '.' is 0, 'a'..'z' are 1..26."""
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos
