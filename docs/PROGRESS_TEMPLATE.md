# Progress Tracker (template)

Copy this to `PROGRESS.md` at the repo root (a fresh clone already has one). The book
runs in one-hour sessions — the map is the [Session Guide](SESSIONS.md), the protocol is
[How to Read This Book](HOW_TO_READ.md). The tutor agent (`/session`) appends one log
entry per session; the file is yours to edit.

## Current Status

**Session:** not started — run `/onboard`, then `/session`
**Profile:** `checkpoints/myllm/PROFILE.md` (written by `/onboard`)
**Milestones green:** `uv run pytest tests/milestones/` is the source of truth

---

## Session Log

One entry per session, appended by `/session` (or by hand — same format):

```markdown
### YYYY-MM-DD — S1.3 (KV-cache budget) — DONE
- **Built**: chose n_kv_heads=2 (GQA); cache 5.2 MB @ batch 8 x 512
- **Done-when**: test_m02_kv_budget.py ✅
- **Sticking point**: per-head vs per-layer cache sizing (ladder rung 2)
- **Next**: S1.4 — dense vs MoE, decided with numbers
```

- `DONE` requires the card's Done-when test green — nothing else counts.
- `PARTIAL` is a normal outcome for a session that hit the hour; it resumes at the
  card's Midpoint next sitting.
- **Sticking point** feeds the next session's two-minute retrieval check — write the
  concept, not just "was stuck".

---

## Cloud Cost Tracker

| Date | Provider | Instance | Hours | Est. Cost |
|------|----------|----------|-------|-----------|
| | | | | |

**Running total:** $0

---

## Key Numbers I've Measured

Fill these in as you go — having your own measurements beats memorizing someone else's.
(Your capstone model's numbers accumulate separately in `checkpoints/myllm/metrics.json`.)

| Metric | tiny | small | medium |
|--------|------|-------|--------|
| Param count | | | |
| Weight memory (FP32) | | | |
| Weight memory (BF16) | | | |
| Training tokens/sec (CPU) | | | |
| Training tokens/sec (A100) | | | |
| MFU (eager FP32) | | | |
| MFU (compile+BF16) | | | |
| Inference tok/s (decode, batch=1) | | | |
| Inference tok/s (prefill, seq=512) | | | |
| INT8 compression ratio | | | |
| KV cache memory (seq=2048, BF16) | | | |
