# Inside a Session

[How to Read This Book](HOW_TO_READ.md) explains the one-hour session in a paragraph;
this page documents the whole machine — what the tutor agent does minute by minute, the
rules it plays by, and where the state lives. It is the reader-facing view of the
contract encoded in the repo's `.claude/skills/` (`session.md`, `onboard.md`,
`stuck.md`), which any clone of the repo ships with. The design rationale lives in
`docs/SESSIONS_SPEC.md` in the repo, for the curious.

The premise, one more time: **what I cannot create, I do not understand.** Everything
below exists to keep the creating on your side of the keyboard.

## The hour at a glance

```
 0:00        0:05                 0:25                       0:55    1:00
  │  Locate   │                    │                          │       │
  │  + Open   │  Read & experiment │          Build           │Verify │ Log
  ├───────────┼────────────────────┼──────────────────────────┼───────┤
  │ agent     │ you read, run,     │ you code; agent reviews, │ run   │ agent
  │ finds the │ predict-then-check │ hints via the ladder,    │ the   │ appends
  │ card,     │ (agent on standby  │ runs tests on request    │ done- │ the log
  │ warms you │  for questions)    │       ▲ Midpoint ▲       │ when  │ entry
  │ up        │                    │  (30-min split point)    │ test  │
```

Three state files drive everything, all outside version control:

| File | Written by | Role |
|------|-----------|------|
| `checkpoints/myllm/PROFILE.md` | `/onboard` (only) | Who you are: background, goal, hardware, pace, hint cap |
| `PROGRESS.md` (repo root) | `/session` appends; you edit freely | The log — one entry per session; doubles as the agent's memory of you |
| `checkpoints/myllm/…` | **you** (only) | The project artifacts the milestone tests inspect |

## The protocol, step by step

**1. Locate.** On `/session` the agent reads your profile, your log, and the
[Session Guide](SESSIONS.md), then finds the current session: the one you name, else the
last `PARTIAL` entry, else the first card in Guide order whose Done-when test doesn't
pass. It checks the card's **Needs** line and, if a prerequisite is missing, offers that
session instead. It also notes the wall-clock time — the timekeeping later is real, not
vibes.

**2. Open** (≤5 min). Two sentences of goal, the Done-when restated — and, if your last
log entry recorded a *Sticking point*, a two-minute retrieval check on exactly that
concept, answered from memory. It's a warm-up, not a gate: cheap spaced repetition,
aimed at the one thing that was hard last time.

**3. Read** (~20 min). You work the card's Read block — chapter section plus notebook
cells, predicting before you check. The agent stays quiet unless asked. It will answer
any question at [Concepts-FAQ](CONCEPTS.md) depth, but it will not summarize the chapter
for you: reading it *is* the session.

**4. Build** (~30 min). You write the milestone step. The agent reviews as you go, asks
pointed questions about suspicious lines, runs the milestone test whenever you want a
check, and hints strictly by the ladder (below). At the card's **Midpoint** — every card
marks one — it offers the stopping point if you're pacing in 30-minute sittings or the
hour is running away.

**5. Verify** (~5 min). The agent runs the card's Done-when command, exactly as printed
on the card. Green: it tells you what the artifact is and what passing proves. Red at
the hour: a normal outcome, logged as `PARTIAL`, resumed at the Midpoint next sitting.
At ~50 minutes the agent proactively suggests moving here.

**6. Log & close** (~5 min). One entry appended to `PROGRESS.md`:

```markdown
### 2026-07-22 — S1.3 (KV-cache budget) — DONE
- **Built**: chose n_kv_heads=2 (GQA); cache 5.2 MB @ batch 8 x 512
- **Done-when**: test_m02_kv_budget.py ✅
- **Sticking point**: per-head vs per-layer cache sizing (ladder rung 2)
- **Next**: S1.4 — dense vs MoE, decided with numbers
```

The *Sticking point* line is the important one — it seeds the next session's retrieval
check, and a repeated theme across entries makes the agent propose a remediation
session before continuing.

## The rules the agent plays by

| The agent will never | The agent will happily |
|---|---|
| Create or edit files under `checkpoints/myllm/` (except `PROFILE.md` via `/onboard`) | Generate **stubs**: signatures, docstrings, `raise NotImplementedError` |
| Write function/method bodies for milestone work — in files *or in chat* | Write, extend, and run **tests**; interpret their failures |
| "Fix" your code by rewriting it | Review your diff and ask what happens on the edge case it spotted |
| Declare a session done without the Done-when test green | Run the test as often as you like and read the failure with you |
| Summarize the chapter so you can skip reading it | Explain any concept, with papers ([Papers](PAPERS.md)) and videos ([Videos](VIDEOS.md)) linked |

If you ask it to break these — "just write it for me", "just update metrics.json" — it
declines and tells you what to write instead. That's the product working, not the
product failing.

## The hint ladder

Stuck is normal and budgeted for. Ask for a hint (or `/stuck`) and the agent climbs
four rungs, one at a time, never skipping ahead uninvited:

1. **Point** — the chapter section, notebook cell, or repo file where the idea lives.
2. **Ask** — one guiding question that isolates the misconception.
3. **Pseudocode** — the shape of the solution, no runnable syntax.
4. **One line** — only on an explicit *"just show me"*: the single blocking line, then
   back to rung 1 for the next obstacle.

A worked example, mid-Build in S1.3 (the KV-cache budget):

> **You**: My cache math says MQA still blows the 8 MB budget. Something's off.
> **Agent** *(rung 1)*: The bytes/token formula is in the chapter's "what the cache
> stores" section — check which factors are per-layer and which are per-head.
> **You**: Still don't see it.
> **Agent** *(rung 2)*: Your `head_dim` — did you compute it from `n_heads`, or from
> `n_kv_heads`?
> **You**: …from `n_kv_heads`. That's it — head_dim is `dim / n_heads` regardless.

The rung you reached is logged. Prefer never seeing rung 4 at all? Tell `/onboard` you
want **hard mode** — the cap goes in your profile and the agent respects it absolutely.

## Done-when: how the tests behave

Every milestone session ends in the same command family:

```bash
uv run pytest tests/milestones/            # the whole project
uv run pytest tests/milestones/test_m02_kv_budget.py   # just today's card
```

The tests read only your workspace, and they have exactly three behaviors:

- **Skip** — the artifact for that session doesn't exist yet. The skip message names
  the session to run. A fresh clone is 26 skips; that's the book, unstarted.
- **Fail** — the artifact exists but doesn't meet the card's acceptance criterion. This
  is a session you attempted and haven't finished, and the failure message says what's
  short.
- **Pass** — done. Not "looks done", not "the agent says it's fine": done.

Because skips are silent-by-design, `uv run pytest` stays green all book long, mid-book
included — the milestone suite only ever *fails* on work you've actually touched.

## Sessions without a milestone

Read-only sessions (S0.1–S0.4, S2.1, S5.1, …) end in a self-check quiz instead of a
test. The cards deliberately list only the quiz *topics* — the agent generates fresh
questions each time, so there's nothing to peek at, and a good answer explains *why*,
not just *what*. Skipping Phase 0 works the same way: `/onboard` quizzes you before
recording the skip.

## No agent? Still a book

Everything above degrades gracefully. The [Session Guide](SESSIONS.md) is a complete
syllabus on its own, the Done-when commands run the same from any terminal, and the log
format is two minutes of typing by hand. The agent adds the tailoring — the retrieval
checks, the ladder discipline, the timekeeping — not the content.
