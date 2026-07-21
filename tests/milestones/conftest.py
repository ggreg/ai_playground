"""Shared fixtures for the milestone acceptance tests.

These tests certify the capstone project's session work (docs/SESSIONS.md). They read
ONLY the reader's workspace (checkpoints/myllm/) and the repo's public APIs. Policy:

- A missing artifact or missing metrics key means the session hasn't been reached yet
  -> pytest.skip, with a message naming the session to run. `uv run pytest` stays green
  for readers mid-book and for CI (which has an empty workspace).
- A present-but-wrong artifact means the session was attempted and its acceptance
  criterion is not met -> the test FAILS. Green means done; nothing else does.
"""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT / "checkpoints" / "myllm"

# Reader-authored modules (e.g. serve_myllm.py from sessions S4.3-S4.5) live in the
# gitignored workspace but must be importable by these tests.
_WS_SRC = WORKSPACE / "src"
if _WS_SRC.is_dir() and str(_WS_SRC) not in sys.path:
    sys.path.insert(0, str(_WS_SRC))


@pytest.fixture
def ws() -> Path:
    if not WORKSPACE.is_dir():
        pytest.skip(
            "Workspace checkpoints/myllm/ not found — start the project at session S0.5 "
            "(or S1.2 if skipping the DNN refresher). See docs/SESSIONS.md."
        )
    return WORKSPACE


def require(path: Path, session: str, what: str) -> Path:
    """Skip (not fail) when an artifact from an earlier/unreached session is absent."""
    if not path.exists():
        pytest.skip(f"{what} not found ({path.name}) — run session {session} first.")
    return path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        pytest.fail(f"{path.name} is not valid JSON: {e}")


@pytest.fixture
def metrics(ws):
    return load_json(require(ws / "metrics.json", "S0.5", "Project metrics"))


def metric(metrics: dict, key: str, session: str):
    """Missing key -> session not reached -> skip. Present key gets validated by the test."""
    if key not in metrics:
        pytest.skip(f"metrics.json has no '{key}' yet — run session {session}.")
    return metrics[key]


@pytest.fixture
def reader_config_raw(ws):
    return load_json(require(ws / "config.json", "S1.2", "Your model's config"))


@pytest.fixture
def reader_config(reader_config_raw):
    """The reader's design as a TransformerConfig.

    config.json legitimately accumulates non-architecture keys over the book
    ("schedule" at M5, "precision" at M6), so unknown keys are filtered, not rejected.
    """
    from ai_playground.models import TransformerConfig

    fields = {f.name for f in dataclasses.fields(TransformerConfig)}
    return TransformerConfig(**{k: v for k, v in reader_config_raw.items() if k in fields})


def load_checkpoint(path: Path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("config", "model"):
        assert key in ckpt, (
            f"{path.name} must be saved as {{'config': cfg_dict, 'model': state_dict}} "
            f"(missing '{key}') — the recipe is in the M4 milestone cell."
        )
    return ckpt


def rebuild_model(ckpt):
    """Reload a checkpoint into a fresh model; key errors here fail the milestone."""
    from ai_playground.models import Transformer, TransformerConfig

    fields = {f.name for f in dataclasses.fields(TransformerConfig)}
    config = TransformerConfig(**{k: v for k, v in ckpt["config"].items() if k in fields})
    model = Transformer(config)
    model.load_state_dict(ckpt["model"], strict=True)
    return model, config


def load_loss_curve(path: Path, session: str) -> list[float]:
    losses = load_json(require(path, session, "Loss curve"))
    assert isinstance(losses, list) and len(losses) >= 2, (
        f"{path.name} must be a JSON list of at least 2 losses, got: {type(losses).__name__}"
    )
    assert all(isinstance(x, (int, float)) for x in losses), (
        f"{path.name} must contain only numbers."
    )
    return losses


def decisions_entry(ws, heading: str, session: str) -> str:
    """Return the text of a `## <heading>` entry in DECISIONS.md."""
    path = require(ws / "DECISIONS.md", session, "Design decisions log")
    text = path.read_text()
    if f"## {heading}" not in text:
        pytest.skip(f"DECISIONS.md has no '## {heading}' entry yet — run session {session}.")
    entry = text.split(f"## {heading}", 1)[1]
    # Entry runs until the next heading (or EOF).
    return entry.split("\n## ", 1)[0]
