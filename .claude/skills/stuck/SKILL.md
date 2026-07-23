---
name: stuck
description: Give a hint on the reader's current obstacle using the book's escalating hint ladder — never the solution. Use when the user says /stuck, "give me a hint", or is blocked on milestone work.
user-invocable: true
---

# Give a Hint

The reader is blocked on project work. Help them get themselves unblocked — the book's
motto is "what I cannot create, I do not understand", so the solution must stay theirs.

1. **Identify the obstacle** from the conversation (and their code, if shared). If it's
   ambiguous, ask one clarifying question — a hint aimed at the wrong obstacle wastes
   the rung.
2. **Pick the lowest useful rung** of the ladder, state which rung you're answering at,
   and answer at exactly that level:
   - **Rung 1 — Point**: the chapter section / notebook cell / repo file where the idea
     lives.
   - **Rung 2 — Ask**: one guiding question that isolates the misconception.
   - **Rung 3 — Pseudocode**: the shape of the solution, no runnable syntax.
   - **Rung 4 — One line**: only on an explicit "just show me", and only if
     `checkpoints/myllm/PROFILE.md` doesn't cap hints below it: the single blocking
     line or expression, nothing around it.
3. **Escalate only when the previous rung demonstrably didn't land** — same obstacle,
   real attempt in between. New obstacle → back to rung 1.

If a /session is running, this counts toward its logged ladder rung. Never write the
implementation, never edit files under `checkpoints/myllm/`, and never "check" their
work by rewriting it — the milestone tests (`uv run pytest tests/milestones/`) are the
verifier.
