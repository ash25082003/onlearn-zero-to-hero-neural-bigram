# Module 0 — Prerequisites (taught inline)

Before the neural bigram model, the learner needs three things checked. Run a quick
**diagnostic** at the very first session, then teach only the warm-ups they actually need.
A confident learner can skip straight to Milestone 1.

## Step 0 — Is PyTorch installed?

Run `python check.py`. If it complains about torch, walk them through `SETUP.md`
(`pip install torch`, CPU is fine) before anything else. This is the only install.

## The diagnostic (ask conversationally, don't make it feel like a test)

Find out, in a friendly back-and-forth:

1. **Did they do Project 1 (the bigram language model)?** This project rebuilds that exact
   model, and its punchline ("same 2.45 loss!") only lands if they know the original. If
   they haven't done it, strongly suggest doing it first; if they insist on starting here,
   do Warm-up A properly.
2. **Bigram model recall** — can they say what `P[i, j]` meant, and why the model's score
   was the *average negative log* of probabilities (not the product)?
3. **Gradient descent intuition** — do they get "the gradient says which way the loss goes
   up, so nudge the weights the other way"? (Module 1 / micrograd makes this deep, but it
   isn't required — PyTorch runs the backward pass here.)

Based on their answers, route them:
- Solid on all → go straight to Milestone 1.
- Shaky on one → do just that warm-up, then continue.
- New to this → warm-ups first, and set expectations warmly (this is still a small,
  finishable project).

## Warm-up A — The bigram model in two minutes (only if needed)
Goal: enough Project 1 context for the punchline to land. A 27x27 table of counts,
rows normalized into "given this character, what comes next?", names sampled by walking
it, quality scored at ~2.454 average NLL. Keep it to a whiteboard-style recap — no code.

## Warm-up B — Gradient descent intuition (only if needed)
Goal: intuition, not rigor. Loss = one number saying how wrong the model is. Gradient =
for each weight, "does the loss rise or fall if I nudge this up?" Training = nudge every
weight a little bit downhill, repeat. One concrete example (one weight, one nudge). Mention
that PyTorch computes all the gradients automatically with `loss.backward()` — that's what
`requires_grad=True` signs the weights up for.

When the needed warm-ups are done (or skipped), continue into the main CURRICULUM.md.
