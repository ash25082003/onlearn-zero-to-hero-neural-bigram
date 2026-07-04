# Curriculum — Train a neural bigram model

Goal: the student rebuilds the bigram model **as a neural network** — one-hot inputs, a
single 27x27 weight matrix, softmax written by hand, negative log-likelihood loss, and a
gradient-descent loop — and watches it converge to the same ~2.45 loss the counting model
achieved in Project 1. By the end they understand that a neural net *learns* what counting
*knew*, and they've built the exact framing (logits -> softmax -> NLL -> backprop) that
scales all the way to GPT.

Reveal these milestones **one at a time**. Do not show the whole list to the student.

## Milestone 1 — The training set
Turn every bigram into a supervised example: `xs[k]` is a character index, `ys[k]` is the
index of the character that follows it. 228,146 examples from 32,033 names.
*Concept:* language modeling as supervised learning — "given this character, predict the
next" is a classification problem with 27 classes.

## Milestone 2 — One-hot & the weight matrix
Encode the integer inputs as one-hot float vectors, and create the network's only layer:
a 27x27 matrix of random weights with `requires_grad=True`.
*Concept:* why ints can't feed a net (index 13 isn't "13 times more" than index 1), and
what `requires_grad` asks PyTorch to do.

## Milestone 3 — Forward pass: softmax by hand
`one_hot @ W` gives logits — interpret them as *log-counts*. Exponentiate to get counts,
normalize each row to get probabilities. That's softmax, written out in three lines. The
`keepdim` trap from Project 1 lives here too.
*Concept:* logits and softmax — how a network outputs a probability distribution while
staying differentiable everywhere.

## Milestone 4 — The loss: NLL + regularization
Pluck out the probability the model gave each *correct* next character, `-log().mean()`
it, and add `reg * (W**2).mean()`. This is exactly Project 1's average NLL — and the
regularization term is exactly its smoothing, wearing a different coat.
*Concept:* one loss, two views: pushing W toward zero blurs the model just like adding
fake counts did.

## Milestone 5 — Train it
The loop: forward, loss, `W.grad = None`, `loss.backward()`, `W.data += -lr * W.grad`.
Watch the loss fall from ~3.8 toward 2.5. (Yes, lr=50 — worth wondering about.)
*Concept:* gradient descent — the same loop as micrograd, but PyTorch does the backward
pass, and it rediscovers the counting model's answer from random weights.

## Milestone 6 — Sample from your network
Project 1's sampling loop, with one change: the row of probabilities comes from a forward
pass instead of a table lookup. The names come out nearly identical — because the network
learned nearly the same distribution.
*Concept:* the 27x27 weight matrix effectively *became* the log of Project 1's count
matrix. Counting and learning converge.

Each milestone has tests in `tests/`. A milestone is **done when its tests pass**.

When all tests pass, run `python train.py` for the full show — then on to the next lecture,
where more context and a real hidden layer beat 2.45 at last (makemore Part 2: MLP).
