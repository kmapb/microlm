"""Where microlm keeps its bytes.

Two roots, deliberately separate:

  * The *data root* holds big, re-downloadable things -- the HuggingFace hub and
    dataset caches, torch hub weights.  On this box that's the shared
    `datasettes` mount, so a dataset downloaded once is there for every run.
  * The *output root* holds things a run produces -- checkpoints and logs.  Those
    are written continuously while training, so they default to local disk.

Import this module *before* `datasets`, `transformers`, or `huggingface_hub`:
those read their cache locations from the environment at import time, and this
module is what sets them.

Nothing here overrides an environment variable that's already set, so
`MICROLM_DATA_ROOT=/somewhere uv run train.py` does what you'd expect, as does
setting `HF_HOME` directly.
"""

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent

# The shared mount on this machine. If it isn't there (laptop, CI, someone
# else's box), fall back to the usual per-user cache instead of failing.
_DATASETTES = Path("/lambda/nfs/datasettes")


def _default_data_root() -> Path:
    if _DATASETTES.is_dir():
        return _DATASETTES / "microlm"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "microlm"


def _env_path(var: str, default: Path) -> Path:
    """Read `var` as a path, defaulting to (and exporting) `default`."""
    value = os.environ.get(var)
    if not value:
        value = str(default)
        os.environ[var] = value
    return Path(value)


DATA_ROOT = _env_path("MICROLM_DATA_ROOT", _default_data_root())
OUTPUT_ROOT = _env_path("MICROLM_OUTPUT_ROOT", _REPO / "runs")

# HF_HOME is the one knob that matters: modern `datasets` and `huggingface_hub`
# put both the hub cache and the dataset cache underneath it.
HF_HOME = _env_path("HF_HOME", DATA_ROOT / "huggingface")
TORCH_HOME = _env_path("TORCH_HOME", DATA_ROOT / "torch")

CHECKPOINT_DIR = OUTPUT_ROOT / "val_ckpts"
LOG_DIR = OUTPUT_ROOT / "logs"


def ensure_dirs() -> None:
    """Create everything we're about to write to. Cheap and idempotent."""
    for d in (DATA_ROOT, HF_HOME, TORCH_HOME, OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    return "\n".join(
        f"  {name:<14} {path}"
        for name, path in [
            ("data root", DATA_ROOT),
            ("hf cache", HF_HOME),
            ("torch cache", TORCH_HOME),
            ("output root", OUTPUT_ROOT),
            ("checkpoints", CHECKPOINT_DIR),
            ("logs", LOG_DIR),
        ]
    )


if __name__ == "__main__":
    ensure_dirs()
    print("microlm paths:")
    print(describe())
