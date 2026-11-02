from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
)

from src.core.modeling import (
    add_calibration,
    build_raw_table,
    feature_sets,
    random_forest,
    select_evaluation_windows,
)


INPUT = Path(
    "results/baseline/xai_oof_predictions.csv"
)

OUTPUT = Path(
    "results/baseline/xai_oof_predictions_calibrated.csv"
)

FEATURE_SET = "personalized_14"
INNER_SPLITS = 5
MIN_STATE_SAMPLES = 100
EPS = 1e-6
N_BINS = 10


def logit_transform(
    values,
):
    values = np.clip(
        np.asarray(
            values,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        values
        / (
            1.0
            - values
        )
    )


def fit_calibrator(
    confidence,
    correct,
):
    confidence = np.asarray(
        confidence,
        dtype=float,
    )

    correct = np.asarray(
        correct,
        dtype=int,
    )

    if len(
        np.unique(
            correct
        )
    ) < 2:
        return None

    x = logit_transform(
        confidence
    ).reshape(
        -1,
        1,
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        x,
        correct,
    )

    return model


def apply_calibrator(
    model,
    confidence,
):
    confidence = np.asarray(
        confidence,
        dtype=float,
    )

    x = logit_transform(
        confidence
    ).reshape(
        -1,
        1,
    )

    return model.predict_proba(
        x
    )[
        :,
        1,
    ]


def predicted_confidence(
    model,
    x,
):
    pred = model.predict(
        x
    )

    proba = model.predict_proba(
        x
    )

    class_to_column = {
        int(
            class_id
        ):
        index
        for index, class_id
        in enumerate(
            model.classes_
        )
    }

    columns = np.asarray(
        [
            class_to_column[
                int(
                    value
                )
            ]
            for value in pred
        ],
        dtype=int,
    )

    confidence = proba[
        np.arange(
            len(
                pred
            )
        ),
        columns,
    ]

    return (
        pred.astype(
            int
        ),
        confidence.astype(
            float
        ),
    )


def inner_oof_predictions(
    x,
    y,
    groups,
):
    unique_groups = np.unique(
        groups
    )

    n_splits = min(
        INNER_SPLITS,
        len(
            unique_groups
        ),
    )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    pred = np.full(
        len(
            y
        ),
        -1,
        dtype=int,
    )

    confidence = np.full(
        len(
            y
        ),
        np.nan,
        dtype=float,
    )

    for (
        train_idx,
        validation_idx,
    ) in splitter.split(
        x,
        y,
        groups,
    ):
        model = random_forest()

        model.fit(
            x[
                train_idx
            ],
            y[
                train_idx
            ],
        )

        fold_pred, fold_confidence = (
            predicted_confidence(
                model,
                x[
                    validation_idx
                ],
            )
        )

        pred[
            validation_idx
        ] = fold_pred

        confidence[
            validation_idx
        ] = fold_confidence

    if np.any(
        pred < 0
    ):
        raise RuntimeError(
            "Missing inner predictions"
        )

    if np.any(
        ~np.isfinite(
            confidence
        )
    ):
        raise RuntimeError(
            "Missing inner confidence"
        )

    return (
        pred,
        confidence,
    )


def calibration_ece(
    correct,
    score,
):
    correct = np.asarray(
        correct,
        dtype=float,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        N_BINS + 1,
    )

    total = len(
        score
    )

    ece = 0.0

    for index in range(
        N_BINS
    ):
        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if index == (
            N_BINS - 1
        ):
            mask = (
                (score >= lower)
                & (score <= upper)
            )
        else:
            mask = (
                (score >= lower)
                & (score < upper)
            )

        n = int(
            mask.sum()
        )

        if n == 0:
            continue

        mean_score = float(
            np.mean(
                score[
                    mask
                ]
            )
        )

        accuracy = float(
            np.mean(
                correct[
                    mask
                ]
            )
        )

        ece += (
            n
            / total
        ) * abs(
            accuracy
            - mean_score
        )

    return float(
        ece
    )


def calibration_metrics(
    correct,
    score,
):
    return {
        "mean_score":
        float(
            np.mean(
                score
            )
        ),

        "accuracy":
        float(
            np.mean(
                correct
            )
        ),

        "bias":
        float(
            np.mean(
                correct
            )
            - np.mean(
                score
            )
        ),

        "brier":
        float(
            brier_score_loss(
                correct,
                score,
            )
        ),

        "ece":
        calibration_ece(
            correct,
            score,
        ),

        "roc_auc":
        float(
            roc_auc_score(
                correct,
                score,
            )
        ),
    }


def selective_accuracy(
    correct,
    score,
    coverage,
):
    n = int(
        np.ceil(
            len(
                score
            )
            * coverage
        )
    )

    order = np.argsort(
        score
    )[
        ::-1
    ]

    selected = order[
        :n
    ]

    return float(
        np.mean(
            correct[
                selected
            ]
        )
    )


def main():
    saved = pd.read_csv(
        INPUT
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

    if len(
        saved
    ) != len(
        evaluation
    ):
        raise RuntimeError(
            "Saved XAI data and evaluation data differ"
        )

    checks = [
        (
            saved[
                "recording"
            ].astype(
                str
            ).to_numpy(),
            evaluation[
                "recording"
            ].astype(
                str
            ).to_numpy(),
            "recording",
        ),
        (
            saved[
                "subject"
            ].astype(
                str
            ).to_numpy(),
            evaluation[
                "subject"
            ].astype(
                str
            ).to_numpy(),
            "subject",
        ),
        (
            saved[
                "window_start_sample"
            ].to_numpy(),
            evaluation[
                "window_start_sample"
            ].to_numpy(),
            "window_start_sample",
        ),
        (
            saved[
                "state"
            ].to_numpy(),
            evaluation[
                "state"
            ].to_numpy(),
            "state",
        ),
    ]

    for left, right, name in checks:
        if not np.array_equal(
            left,
            right,
        ):
            raise RuntimeError(
                f"Mismatch in {name}"
            )

    features = feature_sets(
        "hr_br"
    )[
        FEATURE_SET
    ]

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

    groups = evaluation[
        "subject"
    ].to_numpy(
        dtype=str
    )

    outer_prediction = saved[
        "prediction"
    ].to_numpy(
        dtype=int
    )

    raw_confidence = saved[
        "confidence"
    ].to_numpy(
        dtype=float
    )

    correct = saved[
        "correct"
    ].to_numpy(
        dtype=int
    )

    calibrated_confidence = np.full(
        len(
            saved
        ),
        np.nan,
        dtype=float,
    )

    splitter = LeaveOneGroupOut()

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            x,
            y,
            groups,
        ),
        start=1,
    ):
        subject = groups[
            test_idx[
                0
            ]
        ]

        print(
            f"fold={fold:02d} "
            f"test={subject} "
            f"windows={len(test_idx)}"
        )

        inner_pred, inner_confidence = (
            inner_oof_predictions(
                x[
                    train_idx
                ],
                y[
                    train_idx
                ],
                groups[
                    train_idx
                ],
            )
        )

        inner_correct = (
            inner_pred
            == y[
                train_idx
            ]
        ).astype(
            int
        )

        global_calibrator = fit_calibrator(
            inner_confidence,
            inner_correct,
        )

        if global_calibrator is None:
            raise RuntimeError(
                "Global calibrator could not be fitted"
            )

        calibrators = {}

        for state in [
            0,
            1,
            2,
        ]:
            mask = (
                inner_pred
                == state
            )

            state_correct = (
                inner_correct[
                    mask
                ]
            )

            if (
                int(
                    mask.sum()
                )
                >= MIN_STATE_SAMPLES
                and len(
                    np.unique(
                        state_correct
                    )
                )
                == 2
            ):
                calibrators[
                    state
                ] = fit_calibrator(
                    inner_confidence[
                        mask
                    ],
                    state_correct,
                )
            else:
                calibrators[
                    state
                ] = global_calibrator

        for state in [
            0,
            1,
            2,
        ]:
            mask = (
                outer_prediction[
                    test_idx
                ]
                == state
            )

            local_indices = test_idx[
                mask
            ]

            if len(
                local_indices
            ) == 0:
                continue

            calibrated_confidence[
                local_indices
            ] = apply_calibrator(
                calibrators[
                    state
                ],
                raw_confidence[
                    local_indices
                ],
            )

    if np.any(
        ~np.isfinite(
            calibrated_confidence
        )
    ):
        raise RuntimeError(
            "Missing calibrated confidence"
        )

    consistency = saved[
        "consistency"
    ].to_numpy(
        dtype=float
    )

    reliability_raw = saved[
        "reliability"
    ].to_numpy(
        dtype=float
    )

    reliability_calibrated = np.sqrt(
        calibrated_confidence
        * consistency
    )

    raw_metrics = calibration_metrics(
        correct,
        raw_confidence,
    )

    calibrated_metrics = calibration_metrics(
        correct,
        calibrated_confidence,
    )

    print("")
    print("=" * 100)
    print(
        "GLOBAL CALIBRATION"
    )
    print("=" * 100)

    rows = []

    for name, metrics in [
        (
            "raw",
            raw_metrics,
        ),
        (
            "calibrated",
            calibrated_metrics,
        ),
    ]:
        rows.append(
            {
                "score":
                name,

                **metrics,
            }
        )

    print(
        pd.DataFrame(
            rows
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("")
    print("=" * 100)
    print(
        "CALIBRATION BY PREDICTED STATE"
    )
    print("=" * 100)

    state_rows = []

    for state, state_name in [
        (
            0,
            "Rest",
        ),
        (
            1,
            "Effort",
        ),
        (
            2,
            "Recovery",
        ),
    ]:
        mask = (
            outer_prediction
            == state
        )

        for score_name, score in [
            (
                "raw",
                raw_confidence,
            ),
            (
                "calibrated",
                calibrated_confidence,
            ),
        ]:
            metrics = calibration_metrics(
                correct[
                    mask
                ],
                score[
                    mask
                ],
            )

            state_rows.append(
                {
                    "state":
                    state_name,

                    "score":
                    score_name,

                    "n_windows":
                    int(
                        mask.sum()
                    ),

                    **metrics,
                }
            )

    print(
        pd.DataFrame(
            state_rows
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("")
    print("=" * 100)
    print(
        "RELIABILITY QUALITY"
    )
    print("=" * 100)

    reliability_rows = []

    for name, score in [
        (
            "raw_confidence",
            raw_confidence,
        ),
        (
            "calibrated_confidence",
            calibrated_confidence,
        ),
        (
            "raw_reliability",
            reliability_raw,
        ),
        (
            "calibrated_reliability",
            reliability_calibrated,
        ),
    ]:
        reliability_rows.append(
            {
                "score":
                name,

                "roc_auc":
                roc_auc_score(
                    correct,
                    score,
                ),

                "coverage_20_accuracy":
                selective_accuracy(
                    correct,
                    score,
                    0.20,
                ),

                "coverage_40_accuracy":
                selective_accuracy(
                    correct,
                    score,
                    0.40,
                ),

                "coverage_60_accuracy":
                selective_accuracy(
                    correct,
                    score,
                    0.60,
                ),

                "coverage_80_accuracy":
                selective_accuracy(
                    correct,
                    score,
                    0.80,
                ),
            }
        )

    reliability_frame = pd.DataFrame(
        reliability_rows
    )

    print(
        reliability_frame.round(
            4
        ).to_string(
            index=False
        )
    )

    output = saved.copy()

    output[
        "calibrated_confidence"
    ] = calibrated_confidence

    output[
        "calibrated_reliability"
    ] = reliability_calibrated

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    print("")
    print(
        "saved="
        "results/baseline/"
        "xai_oof_predictions_calibrated.csv"
    )


if __name__ == "__main__":
    main()
