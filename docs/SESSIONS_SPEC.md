# Spec: Interactive Sessions — pairing the book with a coding agent

Status: **implemented** (R1–R4, 2026-07-20) · Author: Greg · Scope: site + notebooks + `.claude/skills/` + `tests/milestones/`

## 1. Motivation

The book (Quarto site + notebooks) and the running project (`docs/PROJECT.md`, milestones
M0–M13) exist, but nothing ties them into a repeatable study rhythm. This spec adds a
**session layer**: the book is consumed in one-hour sessions, each pairing reading and
experimentation with a hands-on project step done alongside a coding agent (Claude Code).

Design principle — *"What I cannot create, I do not understand."* The reader writes every
line of milestone code. The agent teaches, scaffolds, reviews, and verifies; it never
authors the solution. "Done" is decided by an executable test, not by anyone's opinion.

## 2. Goals

1. Every chapter is split into **sessions completable in ~1 hour** (with a marked midpoint
   so a session can also be done as two 30-minute sittings).
2. A fixed session shape: **Read & experiment (~20 min) → Build (~30 min) → Verify & log
   (~10 min)**.
3. The coding agent acts as a **tutor/pair, never an author**, enforced by project skills
   shipped in `.claude/skills/` — any reader who clones the repo gets the same tutor.
4. **Machine-checkable completion**: each session's "done-when" is a pytest in
   `tests/milestones/` (or an in-notebook assert it wraps).
5. **Tailoring to the reader** via a per-reader profile and a session log the agent reads
   at the start of every session and appends to at the end.

## 3. Non-goals

- No server-side state, accounts, or telemetry. All per-reader state lives in the
  gitignored `checkpoints/myllm/` workspace and `PROGRESS.md`.
- No changes to the teaching code in `src/ai_playground/` (except test additions).
- No rewriting of notebook content; only sectioning markers and session cards are added.
- Not a general tutoring framework — this is specific to this book and its one project.

## 4. Vocabulary

| Term | Meaning |
|---|---|
| **Session** | The unit of study: ~1 hour, one card in `docs/SESSIONS.md`, id `S<module>.<k>` after the notebook module directories (`notebooks/02_…` → S2.x; e.g. `S2.4`). |
| **Milestone step** | The buildable slice of a session. Either a whole milestone (M2) or a sub-milestone (M13b). |
| **Workspace** | `checkpoints/myllm/` — gitignored, per-reader. Holds project artifacts, `PROFILE.md`, and all reader-authored milestone code. |
| **Done-when** | The executable acceptance check for a session: `uv run pytest tests/milestones/test_<step>.py`. |
| **Hint ladder** | The only escalation path the agent may use when the reader is stuck (§8). |

## 5. Deliverables

| # | Artifact | Purpose |
|---|---|---|
| D1 | `docs/SESSIONS.md` | The syllabus: one session card per session. Single source of truth for reader and agent. Rendered on the site. |
| D2 | Milestone splits in `docs/PROJECT.md` | M4 and M13 split into hour-sized sub-milestones (§6.2). |
| D3 | `tests/milestones/` | One acceptance test per milestone step (§7). |
| D4 | `.claude/skills/session.md` | The `/session` tutor skill (§8). |
| D5 | `.claude/skills/onboard.md` | The one-time `/onboard` interview skill (§9). |
| D6 | `.claude/skills/stuck.md` | The `/stuck` hint-ladder skill (§8.3). |
| D7 | Log format in `PROGRESS.md` + `docs/PROGRESS_TEMPLATE.md` | Structured session log the agent appends (§10). |
| D8 | Site page "How to read this book" (`docs/HOW_TO_READ.md`) | Explains the session protocol and how to set up Claude Code; linked from the sidebar and `index.qmd`. |

## 6. The syllabus — `docs/SESSIONS.md`

### 6.1 Session card format

One card per session, in book order. Cards are both human-readable and trivially parseable
(stable field labels, one per line). Format:

```markdown
## S3.2 — KV-cache budget: choose your n_kv_heads

- **Chapter**: Attention Mechanisms (`notebooks/01_transformer_internals/01_attention_mechanisms.ipynb`)
- **Read** (~20 min): site §"The KV cache" through §"GQA/MQA tradeoffs"; run notebook cells 12–19, redo the cache-size table for seq_len=4096.
- **Build** (~30 min): milestone **M2** — compute your model's KV-cache budget at your target context length, pick `n_kv_heads`, update `checkpoints/myllm/config.json`, record the reasoning in `DECISIONS.md`.
- **Midpoint**: after the Read block. Stop here if splitting into two 30-min sittings.
- **Done-when**: `uv run pytest tests/milestones/test_m02_kv_budget.py`
- **Needs**: S3.1 (M1 `config.json` exists).
- **Artifact**: updated `checkpoints/myllm/config.json`.
```

Rules:

- Every session has exactly one **Done-when** command and at most one milestone step.
  Read-only sessions (no milestone that hour) are allowed; their Done-when is a short
  self-check quiz block in the card, and the agent administers it.
- **Needs** lists session ids, not prose, so the agent can check prerequisites mechanically
  (each named session's Done-when test must pass, or the artifact it produces must exist).
- Sizing rule of thumb: one session changes **one artifact** in the workspace or turns
  **one assert** green. If a draft card does both, split it.
- `docs/SESSIONS.md` is added to the `_quarto.yml` render list and the sidebar (under
  "🏗️ The Project").

### 6.2 Milestone splits

Existing milestones that fit an hour keep their ids (M0–M3, M5, M6, M8–M12). Oversized
ones split; `docs/PROJECT.md`'s table is updated accordingly:

| New id | Slice | Done-when sketch |
|---|---|---|
| M4a | Wire *your* `config.json` into `scripts/train.py`; 20-step smoke run on CPU/MPS; loss decreases | test asserts a loss-curve JSON exists with `loss[‑1] < loss[0]` |
| M4b | The real 300-step run (Colab/GPU or long CPU run **between** sessions); analyze the curve | `step300.pt` loads; final loss below stated threshold |
| M7a | Effective-batch-size decision + `DECISIONS.md` entry | decision recorded with numbers |
| M7b | Release training run (launched in-session, finishes offline) + verify | `final.pt` loads and generates |
| M13a | Map `final.pt` weights onto the mini-vLLM `MiniLayer` | shapes assert |
| M13b | Greedy parity: paged engine ≡ full recompute on 3 prompts | equality assert |
| M13c | 16-request continuous batching with forced preemption | preemption counter ≥ 1, all requests complete |
| M13d | Decode and read your model's text; final `metrics.json` retrospective | text artifact + all prior `metrics.json` keys present |

Long-running compute (M4b, M7b) is explicitly **launched during a session, completed
between sessions**; the next session's card starts by inspecting the result. A session
never waits on a training run.

## 7. Acceptance tests — `tests/milestones/`

- One file per milestone step: `tests/milestones/test_m02_kv_budget.py`,
  `test_m13b_parity.py`, …
- Tests read **only** the workspace (`checkpoints/myllm/`) and the repo's public APIs.
  They import reader code from the workspace, never from `src/ai_playground/`.
- Tests must **skip with a helpful message** (not fail) when their prerequisite artifacts
  don't exist yet (`pytest.skip("Run session S3.1 first: config.json not found")`), so
  `uv run pytest` stays green for readers mid-book and for CI.
- Each test's docstring states, in one paragraph, *what understanding it certifies* — it
  doubles as the session's exit criterion in prose.
- Tests check outcomes, not implementation style (e.g. "greedy tokens equal reference",
  never "uses a dict for the block table").
- CI runs `tests/milestones/` with an empty workspace and must see all-skipped.

## 8. The tutor — `.claude/skills/session.md`

User-invocable (`/session`, optional arg: a session id to jump to). This skill is the
contract that makes the motto enforceable.

### 8.1 Ground rules (verbatim in the skill)

The agent **must not**:

- Create or edit any file under `checkpoints/myllm/` except `PROFILE.md` and appending to
  `metrics.json` bookkeeping the card explicitly assigns to it — milestone code and
  artifacts are reader-authored.
- Write function/method bodies for milestone work anywhere, including in chat, unless the
  reader has explicitly reached rung 4 of the hint ladder (§8.2) — and then at most one
  focused line or expression.
- Declare a session done without its Done-when test passing.

The agent **may**:

- Write **stubs**: signatures + docstrings + `raise NotImplementedError` / `TODO`, on request.
- Write or extend **tests**, run them, and interpret failures.
- Review the reader's diff and ask questions about it.
- Explain any concept (delegating structure to the existing `/explain` skill) and link
  papers/videos per repo conventions.
- Keep time and manage the log.

### 8.2 Hint ladder

Four rungs, always in order, never skipping ahead uninvited:

1. **Point**: name the chapter section / notebook cell / repo file that contains the idea.
2. **Ask**: one guiding question that isolates the misconception.
3. **Pseudocode**: the shape of the solution, no runnable syntax.
4. **One line**: on explicit reader request only ("just show me"), the single blocking
   line or expression — then back to rung 1 for the next obstacle.

The rung reached is recorded in the session log (§10); repeated rung-3/4 sessions are a
tailoring signal (§9).

### 8.3 `/stuck`

A thin skill that invokes the ladder directly outside a running session: identify the
current obstacle from the conversation, state which rung it is answering at, and answer at
that rung.

### 8.4 Session protocol (state machine)

On `/session`:

1. **Locate** — read `PROGRESS.md`, `checkpoints/myllm/PROFILE.md`, `docs/SESSIONS.md`.
   Current session = the arg if given, else first session whose Done-when doesn't pass.
   Verify **Needs**; if unmet, say so and offer the prerequisite session instead.
2. **Open** (≤5 min) — announce session id and goal, restate Done-when, note the time. If
   the last log entry has a *Sticking point*, run a ≤2-minute retrieval check on it
   (one or two questions, reader answers from memory).
3. **Read** (~20 min) — reader works the card's Read block; agent stands by for questions,
   answering per `/explain` conventions. Agent does not lecture unprompted.
4. **Build** (~30 min) — reader codes; agent reviews, hints (ladder), runs tests on
   request. At the card's **Midpoint** the agent offers a stopping point if pace demands.
5. **Verify** (~5 min) — run the Done-when test. Green → celebrate concretely (name the
   artifact and what it proves). Red at the hour → normal outcome, not failure: log it and
   plan resumption.
6. **Log & close** (~5 min) — append the log entry (§10), preview the next session in one
   sentence. At ~50 minutes of wall clock the agent proactively suggests moving to Verify.

Timekeeping is by wall-clock timestamps the agent notes at Open (e.g. via `date`), not
guesswork.

## 9. Tailoring — `/onboard` and `PROFILE.md`

`/onboard` (run once, re-runnable to revise) interviews the reader — background (math, ML,
PyTorch), goal for the book, hardware (GPU? Colab?), time budget (30 vs 60-minute
sittings), preferred hint style — and writes `checkpoints/myllm/PROFILE.md`:

```markdown
# Reader Profile
- **Background**: expert Python; some PyTorch; no prior CUDA
- **Goal**: understand inference serving end-to-end
- **Hardware**: MacBook (MPS); Colab for GPU-flagged milestones
- **Pace**: 60-minute sessions, weekday evenings
- **Hints**: prefer rung-2 questions over pseudocode
- **Skips**: Phase 0 (self-assessed fluent; verified by quiz on 2026-07-21)
```

The tutor reads this at every Open and adapts: which sessions to propose skipping (Phase 0
is already optional per `docs/LEARNING_PATH.md` — skipping requires passing that phase's
self-check quiz), CPU-fallback vs Colab paths for GPU-flagged milestones, hint-ladder
default rung, and session-splitting at midpoints. Repeated rung-3/4 log entries on a topic
⇒ the agent proposes an extra remediation session (re-derive the concept in a scratch
notebook) before continuing.

`PROFILE.md` lives in the workspace: per-reader, gitignored, no repo pollution.

## 10. Session log — `PROGRESS.md`

`PROGRESS.md` and `docs/PROGRESS_TEMPLATE.md` are restructured around sessions (replacing
the current day-based checklists, which predate the session layer). The agent appends one
entry per session; the reader may edit freely.

```markdown
### 2026-07-22 — S3.2 (KV-cache budget) — DONE
- **Built**: chose n_kv_heads=2 (GQA); config.json updated; cache = 96 MiB @ 4096 ctx
- **Done-when**: test_m02_kv_budget.py ✅
- **Sticking point**: confused per-head vs per-layer cache sizing (ladder rung 2)
- **Next**: S3.3 — MoE decision (M3)
```

Header line: date, session id, `DONE` / `PARTIAL`. `Sticking point` feeds the next
session's retrieval check; `ladder rung N` feeds tailoring. A `PARTIAL` session resumes at
its midpoint next time.

## 11. Site integration

- `docs/SESSIONS.md` and `docs/HOW_TO_READ.md` added to `_quarto.yml` render list and
  sidebar.
- Each chapter page gets a small session banner at the top of each section boundary:
  `⏱️ Session S3.2 starts here — see the session card`, linking into `SESSIONS.md`.
  Implemented as a markdown callout in the notebook's first cell of the section; no Quarto
  extension required.
- `docs/HOW_TO_READ.md` covers: the session shape, installing Claude Code, `uv sync`,
  running `/onboard` then `/session`, and the motto — including *why* the agent will
  refuse to write the reader's milestone code.

## 12. Acceptance criteria for this spec's implementation

1. `docs/SESSIONS.md` exists with cards covering every milestone step in §6.2's table plus
   all unsplit milestones; every card has all seven fields from §6.1.
2. `uv run pytest tests/milestones/` on a fresh clone (empty workspace) reports **0 failed,
   all skipped**, each skip message naming the prerequisite session.
3. `/onboard` then `/session` in a fresh clone reaches the Open state of S0.1 (or the
   first post-quiz session) with no manual setup beyond `uv sync --extra dev`.
4. During a `/session` Build phase, asking the agent "just write it for me" yields a
   ladder response, not an implementation (spot-check against the skill's ground rules).
5. Completing a session appends a well-formed §10 entry and the next `/session` invocation
   locates the following card without arguments.
6. Site renders with the two new pages and at least the Phase 1 chapters carrying session
   banners.

## 13. Rollout plan

| Phase | Content | Why first |
|---|---|---|
| R1 | D1 (`SESSIONS.md`, Phases 0–1 only) + D4 (`session.md`) + D7 (log format) | The loop becomes usable end-to-end on real material immediately. |
| R2 | D3 tests for M0–M3 + D5 (`onboard.md`) + D6 (`stuck.md`) | Done-when becomes executable; tailoring starts. |
| R3 | D2 milestone splits (M4, M7, M13) + remaining session cards + tests | Covers the long-compute and finale sessions. |
| R4 | D8 site page + chapter banners + `_quarto.yml` wiring | Public-facing polish once the mechanics are proven. |

Each phase is dogfooded for at least a week of real sessions before the next begins;
sticking points found while dogfooding amend this spec before they amend the code.

## 14. Resolved questions (decisions as implemented)

1. **Hard mode**: yes — `PROFILE.md` can cap the hint ladder (e.g. "no rung 4");
   `/onboard` offers it, `session.md` and `stuck.md` respect it absolutely.
2. **Reader-authored module code** lives in `checkpoints/myllm/src/` (gitignored);
   `tests/milestones/conftest.py` adds it to `sys.path`. Used by the finale
   (`serve_myllm.py`, M13a–c).
3. **Quizzes are spoiler-free**: session cards list topics only; the agent generates
   the questions each time (see session.md § Tailoring).

Two further conventions set during implementation: loss curves are JSON lists of floats
(`loss_smoke.json`, `loss_step300.json`, `loss_final.json`), and milestones whose
notebook cells predate the session layer carry an appended "Session(s) …" note naming
the exact files/metrics keys the tests check — the session cards and tests are the
authoritative contract.
