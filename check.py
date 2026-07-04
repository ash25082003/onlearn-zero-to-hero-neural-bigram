#!/usr/bin/env python3
"""
check.py - your progress scoreboard for the neural bigram model.

Run it any time from the project root:

    python check.py

It runs your milestones in order and stops at the first one that isn't finished,
so you only ever see what you're working on now - not a wall of red for
milestones you haven't reached yet. Completed milestones stay green; everything
ahead shows as locked until you get there.

The only dependency is PyTorch (see SETUP.md); the tests themselves run on
Python's built-in test tools.
"""
import sys
import unittest
import warnings

# torch prints a harmless warning if numpy isn't installed - not our problem here
warnings.filterwarnings("ignore", message=".*NumPy.*")

try:
    import torch  # noqa: F401
except ImportError:
    print()
    print("  PyTorch isn't installed (or your virtual environment isn't active).")
    print("  This project needs it - it's the one thing to install:")
    print()
    print("      pip install torch")
    print()
    print("  See SETUP.md for the full setup, then run `python check.py` again.")
    print()
    sys.exit(1)

# Ordered milestones: number, title, test module, and a warm hint shown only
# when this is the milestone you're currently on.
MILESTONES = [
    (1, "The training set", "tests.test_milestone1",
     "Walk every name wrapped in '.' tokens; for each bigram append the first "
     "character's index to xs and the second's to ys, then return both as "
     "torch tensors."),
    (2, "One-hot & the weights", "tests.test_milestone2",
     "encode: torch.nn.functional.one_hot(xs, num_classes=27).float(). "
     "init_weights: torch.randn((27, 27), generator=generator, "
     "requires_grad=True)."),
    (3, "Forward pass: softmax by hand", "tests.test_milestone3",
     "logits = encode(xs) @ W, counts = logits.exp(), then divide each ROW by "
     "its sum (keepdim=True!). That's softmax, written out."),
    (4, "The loss (NLL + regularization)", "tests.test_milestone4",
     "probs[torch.arange(n), ys] plucks out the probability of each correct "
     "next character. -log().mean() of that, plus reg * (W**2).mean()."),
    (5, "Train it", "tests.test_milestone5",
     "Each step: forward, loss, W.grad = None, loss.backward(), then "
     "W.data += -lr * W.grad. The loss should fall from ~3.8 toward 2.5."),
    (6, "Sample from your network", "tests.test_milestone6",
     "Same loop as Project 1's sampler, but the row of probabilities now comes "
     "from a forward pass on the current character instead of a lookup table."),
]

GLYPH = {"done": "✅", "current": "▶ ", "locked": "\U0001f512"}  # ✅ ▶ 🔒


def run_module(module_name):
    """Run one milestone's tests. Returns (passed, total, first_problem_detail)."""
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(module_name)
    except Exception as exc:  # tests couldn't even import (e.g. broken neural.py)
        return 0, 1, f"could not load tests ({exc})"
    result = unittest.TestResult()
    suite.run(result)
    total = result.testsRun
    problems = result.failures + result.errors
    passed = total - len(problems)
    detail = None
    if problems:
        lines = problems[0][1].strip().splitlines()
        detail = lines[-1] if lines else "a check failed"
    return passed, total, detail


def main():
    statuses = []  # each: (num, title, state, passed, total, hint, detail)
    blocked = False
    for num, title, module, hint in MILESTONES:
        if blocked:
            statuses.append((num, title, "locked", 0, 0, hint, None))
            continue
        passed, total, detail = run_module(module)
        if total > 0 and passed == total:
            statuses.append((num, title, "done", passed, total, hint, None))
        else:
            statuses.append((num, title, "current", passed, total, hint, detail))
            blocked = True

    done_count = sum(1 for s in statuses if s[2] == "done")
    total_count = len(statuses)

    print()
    print("=" * 62)
    print("  Neural bigram model - your progress")
    print("=" * 62)
    for num, title, state, passed, total, hint, detail in statuses:
        line = f"  {GLYPH[state]}  Milestone {num}  {title}"
        if state == "current":
            extra = f"{passed} of {total} checks passing" if total else "in progress"
            line += f"   <- you are here ({extra})"
        print(line)
    print("-" * 62)

    if not blocked:
        print(f"  \U0001f389  All {total_count} milestones complete - your network learns!")
        print("      Run `python train.py` for the full show: the loss falling")
        print("      to ~2.48 and your network inventing names.")
        print("=" * 62)
        print()
        return 0

    current = next(s for s in statuses if s[2] == "current")
    print(f"  {done_count} of {total_count} milestones complete.")
    print()
    print(f"  ▶  Milestone {current[0]} - {current[1]}: what to do next")
    print(f"     {current[5]}")
    if current[6]:
        print(f"     Failing check: {current[6]}")
    print("=" * 62)
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
