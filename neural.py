"""
neural.py - your neural bigram model.

This is YOUR file. You'll rebuild the bigram model as a tiny neural network here,
one milestone at a time, with the tutor guiding you. The dataset helpers in
`data.py` are already complete - you don't edit them; you use them.

Check your progress any time with:  python check.py
"""
import torch


def build_dataset(words, stoi):
    # TODO (Milestone 1): turn every bigram into a training example. Return two
    # integer tensors (xs, ys) of equal length: xs[k] is a character index and
    # ys[k] is the index of the character that follows it. Wrap each name in
    # the '.' token on both ends, exactly like Project 1.
    raise NotImplementedError("Milestone 1: build_dataset")


def encode(xs):
    # TODO (Milestone 2): one-hot encode a tensor of character indices into a
    # float tensor of shape (len(xs), 27). Integers can't feed a neural net -
    # each index becomes a row of 26 zeros and a single 1.
    raise NotImplementedError("Milestone 2: encode")


def init_weights(generator=None):
    # TODO (Milestone 2): create the network's only layer: a 27x27 weight
    # matrix of random normals (pass generator through!), with
    # requires_grad=True so PyTorch can backpropagate into it.
    raise NotImplementedError("Milestone 2: init_weights")


def forward(xs, W):
    # TODO (Milestone 3): the forward pass, i.e. softmax written by hand.
    # one-hot encode xs, matrix-multiply by W to get logits (log-counts),
    # exponentiate to get counts, then normalize each ROW into probabilities.
    # Return the (len(xs), 27) tensor of probabilities.
    raise NotImplementedError("Milestone 3: forward")


def loss_fn(probs, ys, W, reg=0.01):
    # TODO (Milestone 4): the loss. Pluck out, for every example, the
    # probability the model gave to the CORRECT next character (ys), take the
    # log, average, and negate. Then add the regularization term:
    # reg * (W**2).mean(). Return a single scalar tensor.
    raise NotImplementedError("Milestone 4: loss_fn")


def sample(W, itos, generator=None):
    # TODO (Milestone 6): sample one name from your trained network. Start at
    # the '.' token (index 0), run a forward pass on just that character, draw
    # the next index with torch.multinomial (pass generator through!), and stop
    # when you draw '.' again. Return the name without the dots.
    raise NotImplementedError("Milestone 6: sample")
