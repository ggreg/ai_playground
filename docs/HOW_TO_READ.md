# How to Read This Book

This is a project-based book, built around one motto:

> **What I cannot create, I do not understand.** — Richard Feynman

Every chapter feeds one capstone: [design, train, and serve your own LLM](PROJECT.md).
You read and experiment on this site and in the notebooks — and then **you** write the
project code, pair-programming with a coding agent that tutors but never types the
solution. Understanding is verified the only way that can't be faked: an acceptance test
that only your code can turn green.

## The one-hour session

The unit of the book is not the chapter but the **session** — about an hour, always the
same shape:

| | Time | What happens |
|---|------|--------------|
| **Read & experiment** | ~20 min | The chapter section + notebook cells for the hour. Run everything; predict before you check. |
| **Build** | ~30 min | You write the session's project step. The agent hints, reviews, and keeps time. |
| **Verify & log** | ~10 min | Run the session's **Done-when** test; log what you built and what was hard. |

The full map — every session, its reading, its milestone step, its test — is the
[Session Guide](SESSIONS.md). Each card marks a **Midpoint**: a clean place to stop if
you prefer two 30-minute sittings. Long training runs never block a session — they're
launched at the end of one and harvested at the start of the next. For the complete
minute-by-minute mechanics — the protocol, the agent's rules, the state files — see
[Inside a Session](SESSION_PROTOCOL.md).

## Setup (once)

```bash
git clone https://github.com/ggreg/ai_playground.git
cd ai_playground
uv sync --extra dev
uv run pytest            # everything should pass; milestone tests all skip — for now
```

Install [Claude Code](https://claude.com/claude-code) and open the repo:

```bash
claude
> /onboard    # 5-minute interview: background, goal, hardware, pace, hint style
> /session    # starts (or resumes) your current session
```

`/onboard` writes your profile to `checkpoints/myllm/PROFILE.md` — the gitignored
workspace where your model, your metrics, and eventually your inference engine live.
The tutor reads it every session to adapt: pace, CPU-vs-Colab paths for the GPU-flagged
milestones, how much hinting you want. No GPU required — every GPU milestone has a
stated CPU fallback, and the GPU chapters run on a free Colab T4.

## The rules of the game

**You write the code.** The agent will explain anything, review everything, generate
stubs and tests, and keep you moving — but it will not write your milestone code, and
it will refuse if you ask. That's not stubbornness; it's the product. A milestone the
agent implements teaches the agent.

**Hints escalate.** Stuck is normal and budgeted for. Say `/stuck` (or just ask) and
the agent climbs a four-rung ladder, one rung at a time: **point** you at the right
section → **ask** the question that isolates the misconception → sketch **pseudocode**
→ and only on an explicit "just show me", **one line** — never more. Prefer never
seeing rung 4? Tell `/onboard` you want hard mode.

**Green means done.** Every milestone session ends with one command:

```bash
uv run pytest tests/milestones/
```

Tests for sessions you haven't reached skip (with a pointer to the right session);
tests for work you've attempted pass or fail on the artifact itself. Nobody — you or
the agent — gets to declare a milestone done by eyeballing it.

**The log is the memory.** Each session ends with an entry in `PROGRESS.md`: what you
built, what stuck. The next session opens with a two-minute retrieval check on exactly
that sticking point — cheap spaced repetition, no flashcards.

## Where to start

- Fresh to MLPs and backprop, or want the warm-up: **S0.1** in the
  [Session Guide](SESSIONS.md).
- Fluent already: tell `/onboard` — it quizzes you briefly and records the Phase 0
  skip. You still do **S0.5** (milestone M0): the finale prints its numbers, and they
  should be yours.
- No agent, no problem: the Session Guide works as a plain syllabus, the tests still
  verify your work, and every card stands on its own. The agent makes it tailored; the
  book makes it complete.
