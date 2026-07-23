---
name: onboard
description: One-time reader interview — writes checkpoints/myllm/PROFILE.md so /session can tailor pace, hints, and hardware paths. Use when the user says /onboard, is new to the book, or wants to revise their profile.
user-invocable: true
---

# Onboard a Reader

Interview the reader and write `checkpoints/myllm/PROFILE.md` (create the directory if
needed — it's the gitignored project workspace). This is the ONE file under
`checkpoints/myllm/` the tutor skills may write. Re-running /onboard revises it.

## The interview

Ask conversationally, a couple of questions at a time — not a form. Cover:

1. **Background** — comfort with Python, PyTorch, backprop math, CUDA. Listen for
   specifics ("I've trained models with HF Trainer" ≠ "I've written a training loop").
2. **Goal** — why this book? (serving? training? GPU work? general depth) This decides
   what to emphasize when they ask "can I skip X".
3. **Hardware** — local GPU? Apple Silicon? Colab acceptable for the GPU-flagged
   milestones (M6, M9, M11, M12, and S5.7/S5.8)?
4. **Pace** — 60-minute sessions, or 30-minute sittings split at each card's Midpoint?
   How many per week (sets expectations, nothing else)?
5. **Hint style** — default ladder behavior, or a cap? Offer **hard mode**: rung 4
   disabled entirely ("no rung 4"), for readers who never want to see a line of solution.

## Phase 0 skip-quiz

If they claim MLPs/backprop/cross-entropy fluency and want to skip Phase 0, administer
a short quiz before granting it — 4 questions, answered from memory, e.g.: why `grad +=`
rather than `=` in backprop; what subtract-max changes in softmax and why it's safe; why
MSE trains classifiers badly; parameter count of `MLP(2, [8, 3])` with biases. 3 of 4
solid → record the skip (S0.5/M0 is still required — the fastest path is doing it cold).
Otherwise recommend starting at S0.1, framed as calibration, not failure.

## PROFILE.md format

```markdown
# Reader Profile
- **Background**: <one line, specific>
- **Goal**: <one line>
- **Hardware**: <local situation; Colab yes/no for GPU milestones>
- **Pace**: <60-min sessions | 30-min sittings>, <n>/week
- **Hints**: <default ladder | no rung 4 (hard mode) | other cap>
- **Skips**: <none | Phase 0 (quiz passed YYYY-MM-DD, S0.5 still due)>
- **Notes**: <anything else that changes tutoring>
```

Close by pointing at the next step: `/session` (starting at S0.1, or S1.1 with a
recorded skip), and where the session map lives (`docs/SESSIONS.md`).
