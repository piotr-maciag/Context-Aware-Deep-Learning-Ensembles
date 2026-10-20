"""Survey-disjoint beach-surface training, validation, and test metadata."""

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


TEST_DATES = {"20180812", "20180915", "20220910"}
VALIDATION_DATES = {"20230731", "20240623"}
SPLIT_POLICY = "bs_survey_disjoint_train_validation_test_v2"
CLASS_CODES = {str(code) for code in range(10)}


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bs_splits(train, validation, test):
    frames = {"train": train, "validation": validation, "test": test}
    for name, frame in frames.items():
        if frame.empty or not frame["split"].eq(name).all():
            raise ValueError(f"Invalid or empty BS {name} split; regenerate BS splits")
        if set(frame["class"].astype(str)) != CLASS_CODES:
            raise ValueError(f"BS {name} must contain all ten classes")
    if set(validation.date.astype(str)) != VALIDATION_DATES:
        raise ValueError("BS validation must contain exactly the two reserved surveys")
    if set(test.date.astype(str)) != TEST_DATES:
        raise ValueError("BS test must contain exactly the three reserved surveys")
    date_sets = [set(frame.date.astype(str)) for frame in frames.values()]
    for index, dates in enumerate(date_sets):
        if any(dates & other for other in date_sets[index + 1:]):
            raise ValueError("BS surveys overlap across training, validation, or testing")
    combined = pd.concat(frames.values(), ignore_index=True)
    if combined.duplicated(["date", "cell_id"]).any():
        raise ValueError("Duplicate BS date/cell_id pairs")


def assign_bs_splits(descriptor):
    frame = descriptor.copy()
    for name in ("date", "cell_id", "class"):
        frame[name] = frame[name].astype(str).str.strip()
    test = frame.loc[frame.date.isin(TEST_DATES)].assign(split="test")
    validation = frame.loc[frame.date.isin(VALIDATION_DATES)].assign(split="validation")
    train = frame.loc[~frame.date.isin(TEST_DATES | VALIDATION_DATES)].assign(split="train")
    validate_bs_splits(train, validation, test)
    return train, validation, test


def save_bs_split_frames(train, validation, test, directory):
    validate_bs_splits(train, validation, test)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    columns = ["date", "cell_id", "class", "split"]
    frames = {"train": train, "validation": validation, "test": test}
    files = {}
    for name, frame in frames.items():
        files[name] = directory / f"bs_{name}_split.csv"
        frame[columns].sort_values(["date", "cell_id"]).to_csv(files[name], index=False)
    files["all"] = directory / "bs_dataset_splits.csv"
    pd.concat(frames.values(), ignore_index=True)[columns].sort_values(
        ["split", "date", "cell_id"]
    ).to_csv(files["all"], index=False)
    provenance = {
        "policy": SPLIT_POLICY,
        "test_dates": sorted(TEST_DATES),
        "validation_dates": sorted(VALIDATION_DATES),
        "csv_sha256": {name: file_sha256(files[name]) for name in frames},
    }
    (directory / "bs_split_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return files, provenance


def current_bs_split_provenance(directory):
    directory = Path(directory)
    provenance = json.loads((directory / "bs_split_provenance.json").read_text())
    if (provenance.get("policy") != SPLIT_POLICY
            or set(provenance.get("test_dates", [])) != TEST_DATES
            or set(provenance.get("validation_dates", [])) != VALIDATION_DATES):
        raise ValueError("Outdated BS split policy; rerun bs_create_dataset_splits")
    for name in ("train", "validation", "test"):
        if file_sha256(directory / f"bs_{name}_split.csv") != provenance["csv_sha256"][name]:
            raise ValueError("BS split files changed; regenerate splits and retrain")
    return provenance
