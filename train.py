"""
train.py - train your neural bigram model and watch it learn.

This is YOUR file. You'll build the training loop here (Milestone 5), then run
`python train.py` to watch the loss fall toward the counting model's 2.454 -
and see your network invent names.
"""
import torch

from data import load_words, build_vocab
from neural import build_dataset, init_weights, forward, loss_fn, sample


def train(xs, ys, W, steps=100, lr=50, reg=0.01, verbose=False):
    # TODO (Milestone 5): the training loop. For each step:
    #   1. forward pass -> probs
    #   2. loss = loss_fn(probs, ys, W, reg)
    #   3. reset the gradient (W.grad = None), then loss.backward()
    #   4. nudge the weights: W.data += -lr * W.grad
    # If verbose, print the loss every 10 steps so you can watch it fall.
    # Return the final loss as a float.
    raise NotImplementedError("Milestone 5: train")


if __name__ == "__main__":
    # The payoff. Works once Milestones 1-6 are all done.
    words = load_words("names.txt")
    stoi, itos = build_vocab(words)
    xs, ys = build_dataset(words, stoi)

    g = torch.Generator().manual_seed(2147483647)
    W = init_weights(g)

    print("training on", xs.nelement(), "bigrams...")
    final = train(xs, ys, W, steps=100, lr=50, verbose=True)
    print(f"\nfinal loss: {final:.4f}   (Project 1's counting model: 2.454)")

    print("\nnames your network dreams up:")
    g2 = torch.Generator().manual_seed(2147483647)
    for _ in range(10):
        print("  ", sample(W, itos, g2) or "(empty)")
