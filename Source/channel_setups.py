"""Named feature-channel ablations shared by training, evaluation and inference."""

import os
from pathlib import Path


CHANNEL_SETUPS = {
    "RGB_only": ("r", "g", "b"),
    "RGB_slope": ("r", "g", "b", "s"),
    "RGB_elevation": ("r", "g", "b", "d"),
    "RGB_slope_elevation": ("r", "g", "b", "s", "d"),
}
DEFAULT_CHANNEL_SETUP = "RGB_slope_elevation"


def get_channel_setup(name=None):
    """Select one setup; batch launchers explicitly iterate all four."""
    if name is None:
        name = os.environ.get("BEACH_CHANNEL_SETUP", DEFAULT_CHANNEL_SETUP)
    if name not in CHANNEL_SETUPS:
        raise ValueError(f"Unknown channel setup {name!r}; choose from {list(CHANNEL_SETUPS)}")
    return name, CHANNEL_SETUPS[name]


def setup_result_dir(base, setup=None):
    name, _ = get_channel_setup(setup)
    return Path(base) / name


def validate_setup_metadata(metadata, setup=None):
    name, suffixes = get_channel_setup(setup)
    if tuple(metadata.get("feature_suffixes", ())) != suffixes:
        raise ValueError(f"Artifact channels do not match selected setup {name}")
    if metadata.get("channel_setup", name) != name:
        raise ValueError(f"Artifact was saved for a different channel setup than {name}")
    if len(metadata.get("channel_mean", ())) != len(suffixes) or len(metadata.get("channel_std", ())) != len(suffixes):
        raise ValueError(f"Normalization statistics do not match channels for {name}")


def require_completed_result(directory):
    if (Path(directory) / "experiment_in_progress.json").exists():
        raise ValueError(f"Incomplete or running experiment in {directory}; rerun it to completion before evaluation or inference")
