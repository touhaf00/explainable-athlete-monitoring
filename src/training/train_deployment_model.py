from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.core.calibration import (
    MIN_STATE_SAMPLES,
    fit_calibrator,
)
from src.core.modeling import (
    add_calibration,
    build_raw_table,
    feature_sets,
    random_forest,
    select_evaluation_windows,
)
from src.core.semantic import (
    STATE_NAMES,
    build_profiles,
)


ARTIFACT_DIR = Path(
    "artifacts/deployment"
)

OOF_INPUT = Path(
    "results/baseline/xai_oof_predictions.csv"
)

FEATURE_SET = "personalized_14"
MODE = "hr_br"

EXPECTED_WINDOWS = 9166
EXPECTED_SUBJECTS = 38
EXPECTED_FEATURES = 14

BASELINE_TAG = "baseline-v1.0"
BASELINE_COMMIT = (
    "6db9dd951b3c81f0cae3fc3334b708c6bbd29cc8"
)


def file_sha256(
    path,
):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def validate_oof_alignment(
    oof,
    evaluation,
):
    if len(oof) != len(
        evaluation
    ):
        raise RuntimeError(
            "OOF and evaluation sizes differ"
        )

    checks = [
        (
            oof[
                "recording"
            ].astype(str).to_numpy(),
            evaluation[
                "recording"
            ].astype(str).to_numpy(),
            "recording",
        ),
        (
            oof[
                "subject"
            ].astype(str).to_numpy(),
            evaluation[
                "subject"
            ].astype(str).to_numpy(),
            "subject",
        ),
        (
            oof[
                "window_start_sample"
            ].to_numpy(),
            evaluation[
                "window_start_sample"
            ].to_numpy(),
            "window_start_sample",
        ),
        (
            oof[
                "window_end_sample"
            ].to_numpy(),
            evaluation[
                "window_end_sample"
            ].to_numpy(),
            "window_end_sample",
        ),
        (
            oof[
                "state"
            ].to_numpy(),
            evaluation[
                "state"
            ].to_numpy(),
            "state",
        ),
    ]

    for (
        left,
        right,
        name,
    ) in checks:
        if not np.array_equal(
            left,
            right,
        ):
            raise RuntimeError(
                f"OOF mismatch: {name}"
            )


def fit_final_calibrators(
    oof,
):
    prediction = oof[
        "prediction"
    ].to_numpy(
        dtype=int
    )

    confidence = oof[
        "confidence"
    ].to_numpy(
        dtype=float
    )

    correct = oof[
        "correct"
    ].to_numpy(
        dtype=int
    )

    global_calibrator = (
        fit_calibrator(
            confidence,
            correct,
        )
    )

    if global_calibrator is None:
        raise RuntimeError(
            "Global calibrator could not be fitted"
        )

    by_state = {}
    source = {}
    counts = {}

    for state in sorted(
        STATE_NAMES
    ):
        mask = (
            prediction
            == state
        )

        state_correct = correct[
            mask
        ]

        count = int(
            mask.sum()
        )

        counts[
            state
        ] = count

        if (
            count
            >= MIN_STATE_SAMPLES
            and len(
                np.unique(
                    state_correct
                )
            )
            == 2
        ):
            calibrator = (
                fit_calibrator(
                    confidence[
                        mask
                    ],
                    state_correct,
                )
            )

            if calibrator is None:
                raise RuntimeError(
                    f"State calibrator failed: {state}"
                )

            by_state[
                state
            ] = calibrator

            source[
                state
            ] = "state"
        else:
            by_state[
                state
            ] = global_calibrator

            source[
                state
            ] = "global"

    return {
        "global":
        global_calibrator,

        "by_state":
        by_state,

        "source":
        source,

        "counts":
        counts,
    }


def main():
    if not OOF_INPUT.exists():
        raise FileNotFoundError(
            OOF_INPUT
        )

    print(
        "Loading deployment training data..."
    )

    raw = build_raw_table(
        MODE
    )

    calibrated = add_calibration(
        raw,
        MODE,
    )

    evaluation = (
        select_evaluation_windows(
            calibrated
        )
    )

    features = feature_sets(
        MODE
    )[
        FEATURE_SET
    ]

    if len(
        evaluation
    ) != EXPECTED_WINDOWS:
        raise RuntimeError(
            f"Expected {EXPECTED_WINDOWS} windows, "
            f"got {len(evaluation)}"
        )

    subject_count = (
        evaluation[
            "subject"
        ].nunique()
    )

    if (
        subject_count
        != EXPECTED_SUBJECTS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} subjects, "
            f"got {subject_count}"
        )

    if (
        len(features)
        != EXPECTED_FEATURES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {len(features)}"
        )

    x = evaluation[
        features
    ].to_numpy(
        dtype=float
    )

    y = evaluation[
        "state"
    ].to_numpy(
        dtype=int
    )

    print(
        f"windows={len(evaluation)}"
    )

    print(
        f"subjects={subject_count}"
    )

    print(
        f"features={len(features)}"
    )

    print(
        "Training final Random Forest..."
    )

    model = random_forest()

    model.fit(
        x,
        y,
    )

    if not np.array_equal(
        model.classes_,
        np.asarray(
            [0, 1, 2]
        ),
    ):
        raise RuntimeError(
            f"Unexpected classes: {model.classes_}"
        )

    print(
        "Loading OOF predictions..."
    )

    oof = pd.read_csv(
        OOF_INPUT
    )

    validate_oof_alignment(
        oof,
        evaluation,
    )

    print(
        "Fitting final confidence calibrators..."
    )

    calibrators = (
        fit_final_calibrators(
            oof
        )
    )

    print(
        "Building semantic profiles..."
    )

    profiles = build_profiles(
        evaluation,
        features,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        ARTIFACT_DIR
        / "random_forest.joblib"
    )

    calibrator_path = (
        ARTIFACT_DIR
        / "confidence_calibrators.joblib"
    )

    profiles_path = (
        ARTIFACT_DIR
        / "semantic_profiles.joblib"
    )

    metadata_path = (
        ARTIFACT_DIR
        / "metadata.json"
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        calibrators,
        calibrator_path,
    )

    joblib.dump(
        profiles,
        profiles_path,
    )

    metadata = {
        "baseline_tag":
        BASELINE_TAG,

        "baseline_commit":
        BASELINE_COMMIT,

        "mode":
        MODE,

        "feature_set":
        FEATURE_SET,

        "features":
        list(
            features
        ),

        "n_features":
        len(
            features
        ),

        "n_windows":
        len(
            evaluation
        ),

        "n_subjects":
        int(
            subject_count
        ),

        "classes":
        {
            str(state):
            name
            for state, name
            in STATE_NAMES.items()
        },

        "model":
        {
            "type":
            "RandomForestClassifier",

            "n_estimators":
            300,

            "max_depth":
            8,

            "min_samples_leaf":
            3,

            "class_weight":
            "balanced",

            "random_state":
            42,
        },

        "confidence_calibration":
        {
            "method":
            "state-conditioned Platt calibration",

            "source":
            "LOSO OOF predictions",

            "minimum_state_samples":
            MIN_STATE_SAMPLES,

            "state_sources":
            {
                str(state):
                value
                for state, value
                in calibrators[
                    "source"
                ].items()
            },

            "state_counts":
            {
                str(state):
                int(value)
                for state, value
                in calibrators[
                    "counts"
                ].items()
            },
        },

        "oof_source":
        str(
            OOF_INPUT
        ),

        "oof_sha256":
        file_sha256(
            OOF_INPUT
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded_model = joblib.load(
        model_path
    )

    prediction_before = (
        model.predict(
            x
        )
    )

    prediction_after = (
        loaded_model.predict(
            x
        )
    )

    if not np.array_equal(
        prediction_before,
        prediction_after,
    ):
        raise RuntimeError(
            "Predictions changed after reload"
        )

    probability_before = (
        model.predict_proba(
            x
        )
    )

    probability_after = (
        loaded_model.predict_proba(
            x
        )
    )

    max_probability_difference = float(
        np.max(
            np.abs(
                probability_before
                - probability_after
            )
        )
    )

    if not np.allclose(
            probability_before,
            probability_after,
            rtol=0.0,
            atol=1e-12,
    ):
        raise RuntimeError(
            "Probabilities changed after reload: "
            f"max_difference={max_probability_difference}"
        )

    loaded_calibrators = (
        joblib.load(
            calibrator_path
        )
    )

    loaded_profiles = (
        joblib.load(
            profiles_path
        )
    )

    if (
        set(
            loaded_calibrators[
                "by_state"
            ]
        )
        != set(
            STATE_NAMES
        )
    ):
        raise RuntimeError(
            "Calibrator reload failed"
        )

    if (
        len(
            loaded_profiles
        )
        != 4
    ):
        raise RuntimeError(
            "Semantic profile reload failed"
        )

    print("")
    print(
        "=" * 80
    )

    print(
        "DEPLOYMENT MODEL READY"
    )

    print(
        "=" * 80
    )

    print(
        f"model={model_path}"
    )

    print(
        f"calibrators={calibrator_path}"
    )

    print(
        f"semantic_profiles={profiles_path}"
    )

    print(
        f"metadata={metadata_path}"
    )

    print(
        "reload_predictions=OK"
    )

    print(
        "reload_probabilities=OK"
    )

    print(
        "max_probability_reload_difference="
        f"{max_probability_difference:.3e}"
    )

    for state in sorted(
        STATE_NAMES
    ):
        print(
            f"{STATE_NAMES[state]}: "
            f"calibration_source="
            f"{calibrators['source'][state]} "
            f"n="
            f"{calibrators['counts'][state]}"
        )


if __name__ == "__main__":
    main()
