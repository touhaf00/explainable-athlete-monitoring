from functools import lru_cache
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.core.calibration import apply_calibrator

from src.experiments.hitl_robust import (
    robust_threshold,
    ACCEPT_TARGET,
    ACCEPT_Q25_TARGET,
    NON_REVIEW_TARGET,
    NON_REVIEW_Q25_TARGET,
    MIN_SPORT_SUBJECTS,
    MIN_SPORT_WINDOWS,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/deployment"


@lru_cache(maxsize=1)
def load_artifacts():
    metadata = json.loads(
        (ARTIFACTS / "metadata.json").read_text()
    )

    model = joblib.load(
        ARTIFACTS / "random_forest.joblib"
    )

    calibrators = joblib.load(
        ARTIFACTS / "confidence_calibrators.joblib"
    )

    profiles = joblib.load(
        ARTIFACTS / "semantic_profiles.joblib"
    )

    assert model.n_features_in_ == 14
    assert metadata["feature_set"] == "personalized_14"
    assert set(model.classes_) == {0, 1, 2}

    return metadata, model, calibrators, profiles


@lru_cache(maxsize=1)
def calibration_reference():
    metadata, _, calibrators, _ = load_artifacts()

    path = ROOT / metadata["oof_source"]

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if digest != metadata["oof_sha256"]:
        raise RuntimeError(
            "Les predictions historiques ne correspondent "
            "pas aux artefacts du modele."
        )

    df = pd.read_csv(path)

    scores = np.full(len(df), np.nan)

    for state in [0, 1, 2]:
        mask = df["prediction"].eq(state).to_numpy()

        scores[mask] = apply_calibrator(
            calibrators["by_state"][state],
            df.loc[mask, "confidence"].to_numpy(),
        )

    if not np.isfinite(scores).all():
        raise RuntimeError(
            "Scores calibres invalides."
        )

    df["deployment_score"] = scores

    return df


@lru_cache(maxsize=16)
def fit_deployment_policy(sport):
    df = calibration_reference()

    local = df[df["sport"] == sport]

    if (
        len(local) >= MIN_SPORT_WINDOWS
        and local["subject"].nunique()
        >= MIN_SPORT_SUBJECTS
    ):
        reference = local
        source = "sport"
    else:
        reference = df
        source = "global_fallback"

    scores = reference["deployment_score"]
    correct = reference["correct"]
    subjects = reference["subject"]

    accept = robust_threshold(
        scores,
        correct,
        subjects,
        ACCEPT_TARGET,
        ACCEPT_Q25_TARGET,
    )

    check = robust_threshold(
        scores,
        correct,
        subjects,
        NON_REVIEW_TARGET,
        NON_REVIEW_Q25_TARGET,
    )

    accept_threshold = float(
        accept["threshold"]
    )

    check_threshold = min(
        float(check["threshold"]),
        accept_threshold,
    )

    return {
        "accept": accept_threshold,
        "check": check_threshold,
        "source": source,
    }


def decide(score, policy):
    if not np.isfinite(score):
        return "Review"

    if score >= policy["accept"]:
        return "Accept"

    if score >= policy["check"]:
        return "Check"

    return "Review"


@lru_cache(maxsize=16)
def get_policy(sport):
    path = ARTIFACTS / "hitl_policy.json"

    if path.exists():
        policies = json.loads(path.read_text())

        policy = policies.get(
            sport,
            policies["OTHER"]
        )

        return {
            "accept": float(policy["accept"]),
            "check": float(policy["check"]),
            "source": policy["source"],
        }

    return fit_deployment_policy(sport)
