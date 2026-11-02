import json
from pathlib import Path

import joblib
import numpy as np

from src.core.calibration import apply_calibrator
from src.core.modeling import (
    add_calibration,
    build_raw_table,
    feature_sets,
    select_evaluation_windows,
)
from src.core.semantic import (
    STATE_NAMES,
    semantic_consistency,
    shap_matrix,
)


ARTIFACT_DIR = Path(
    "artifacts/deployment"
)


def main():
    metadata = json.loads(
        (
            ARTIFACT_DIR
            / "metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert metadata[
        "baseline_tag"
    ] == "baseline-v1.0"

    assert metadata[
        "n_windows"
    ] == 9166

    assert metadata[
        "n_subjects"
    ] == 38

    assert metadata[
        "n_features"
    ] == 14

    model = joblib.load(
        ARTIFACT_DIR
        / "random_forest.joblib"
    )

    calibrators = joblib.load(
        ARTIFACT_DIR
        / "confidence_calibrators.joblib"
    )

    profiles = joblib.load(
        ARTIFACT_DIR
        / "semantic_profiles.joblib"
    )

    raw = build_raw_table(
        "hr_br"
    )

    calibrated = add_calibration(
        raw,
        "hr_br",
    )

    evaluation = select_evaluation_windows(
        calibrated
    )

    features = feature_sets(
        "hr_br"
    )[
        "personalized_14"
    ]

    sample = evaluation.iloc[
        [0]
    ]

    x = sample[
        features
    ].to_numpy(
        dtype=float
    )

    prediction = int(
        model.predict(
            x
        )[0]
    )

    probabilities = (
        model.predict_proba(
            x
        )[0]
    )

    class_position = int(
        np.where(
            model.classes_
            == prediction
        )[0][0]
    )

    raw_confidence = float(
        probabilities[
            class_position
        ]
    )

    calibrated_confidence = float(
        apply_calibrator(
            calibrators[
                "by_state"
            ][
                prediction
            ],
            np.asarray(
                [
                    raw_confidence
                ]
            ),
        )[0]
    )

    shap_values = shap_matrix(
        model,
        x,
    )

    class_shap = shap_values[
        0,
        :,
        class_position,
    ]

    consistency, details = (
        semantic_consistency(
            sample.iloc[
                0
            ],
            prediction,
            class_shap,
            features,
            str(
                sample.iloc[
                    0
                ][
                    "sport"
                ]
            ),
            profiles,
        )
    )

    assert prediction in STATE_NAMES
    assert 0.0 <= raw_confidence <= 1.0
    assert 0.0 <= calibrated_confidence <= 1.0
    assert 0.0 <= consistency <= 1.0
    assert len(
        class_shap
    ) == 14

    assert (
        len(details)
        <= 5
    )

    print(
        f"state={STATE_NAMES[prediction]}"
    )

    print(
        f"raw_confidence={raw_confidence:.6f}"
    )

    print(
        "calibrated_confidence="
        f"{calibrated_confidence:.6f}"
    )

    print(
        "semantic_consistency="
        f"{consistency:.6f}"
    )

    print(
        "DEPLOYMENT ARTIFACT TEST: OK"
    )


if __name__ == "__main__":
    main()
