# Train a neural bigram model

**Neural Networks: Zero to Hero - Module 2 - Project 2**
Companion lecture: [The spelled-out intro to language modeling: building makemore](https://youtu.be/PaCmpygFfXo) by Andrej Karpathy (second half).

In Project 1 you built a language model by **counting**. Now build the same model as a
**neural network** - one-hot inputs, a single 27x27 weight matrix, softmax written by hand,
and a gradient-descent loop - and watch it converge to the same ~2.45 loss your counting
model achieved. The network *learns* what counting *knew*. This framing (logits -> softmax ->
NLL -> backprop) is exactly what scales up to GPT.

You'll build every piece yourself, with an AI tutor guiding you one step at a time. PyTorch
handles the backward pass; you write everything else.

## What you need
- Python 3.9+ and **PyTorch** - the one thing to install: `pip install torch` (CPU is
  plenty). Progress checks run on Python's built-in `unittest` via `python check.py`.
- Any coding agent or AI assistant that can read the project files and run shell commands
  (Claude, Codex, Cursor, or another tool)
- Ideally, **Project 1 - Build the bigram language model** under your belt: this project
  rebuilds that exact model, and the punchline lands harder if you built the original.

## How to start
1. Download or clone this project and `cd` into it.
2. Open it with your preferred coding agent.
3. Just say hi. Your tutor takes it from there.

For tool-specific setup notes, see [SETUP.md](./SETUP.md). It covers Claude Code, Codex,
Cursor, and generic coding assistants.

You'll have a network training in about an hour. Your code goes in `neural.py` and
`train.py` (the dataset helpers in `data.py` are provided - you wrote them in Project 1).
Ask for a hint when stuck, ask the tutor to check your work to run the tests, and ask to
move on when ready for the next milestone.

The tests decide when you're done - not the tutor. When they all pass, run `python train.py`
and watch the loss fall while your network invents names.

Next up: **[Module 3 - Build an MLP language model](https://onlearn.app/projects/nn-zero-to-hero/mlp-language-model)**,
where more context and a real hidden layer finally beat 2.45.

---

## More from Onlearn

This is **Module 2 - Project 2** of **[Neural Networks: Zero to Hero](https://onlearn.app/projects/nn-zero-to-hero)**
on Onlearn — guided, build-it-yourself projects with an AI tutor that checks your work against
the tests.

**[→ Explore more courses](https://onlearn.app/projects)**
