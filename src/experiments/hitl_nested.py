from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut

from src.core.calibration import (
    apply_calibrator,
    fit_calibrator,
    inner_oof_predictions,
)

from src.core.modeling import (
    add_calibration,
    build_raw_table,
    feature_sets,
    select_evaluation_windows,
)


INPUT = Path(
    "results/baseline/xai_oof_predictions_calibrated.csv"
)

OUTPUT_DIR = Path(
    "results/experiments/hitl"
)

FEATURE_SET = "personalized_14"

ACCEPT_TARGET = 0.95
NON_REVIEW_TARGET = 0.85

MIN_STATE_SAMPLES = 100


def fit_state_calibrators(
    confidence,
    prediction,
    correct,
):
    global_calibrator = fit_calibrator(
        confidence,
        correct,
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
            prediction
            == state
        )

        state_correct = correct[
            mask
        ]

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
                confidence[
                    mask
                ],
                state_correct,
            )
        else:
            calibrators[
                state
            ] = global_calibrator

    return calibrators


def apply_state_calibrators(
    confidence,
    prediction,
    calibrators,
):
    calibrated = np.full(
        len(
            confidence
        ),
        np.nan,
        dtype=float,
    )

    for state in [
        0,
        1,
        2,
    ]:
        mask = (
            prediction
            == state
        )

        if not np.any(
            mask
        ):
            continue

        calibrated[
            mask
        ] = apply_calibrator(
            calibrators[
                state
            ],
            confidence[
                mask
            ],
        )

    if np.any(
        ~np.isfinite(
            calibrated
        )
    ):
        raise RuntimeError(
            "Missing calibrated training confidence"
        )

    return calibrated


def threshold_for_target(
    score,
    correct,
    target,
):
    score = np.asarray(
        score,
        dtype=float,
    )

    correct = np.asarray(
        correct,
        dtype=int,
    )

    order = np.argsort(
        score
    )[
        ::-1
    ]

    sorted_score = score[
        order
    ]

    sorted_correct = correct[
        order
    ]

    cumulative_correct = np.cumsum(
        sorted_correct
    )

    cumulative_n = np.arange(
        1,
        len(
            sorted_correct
        )
        + 1,
    )

    cumulative_precision = (
        cumulative_correct
        / cumulative_n
    )

    boundary = np.ones(
        len(
            sorted_score
        ),
        dtype=bool,
    )

    boundary[
        :-1
    ] = (
        sorted_score[
            :-1
        ]
        != sorted_score[
            1:
        ]
    )

    valid = np.where(
        boundary
        & (
            cumulative_precision
            >= target
        )
    )[0]

    if len(
        valid
    ) == 0:
        return (
            np.inf,
            0,
            np.nan,
        )

    index = int(
        valid[
            -1
        ]
    )

    threshold = float(
        sorted_score[
            index
        ]
    )

    n_selected = int(
        index
        + 1
    )

    precision = float(
        cumulative_precision[
            index
        ]
    )

    return (
        threshold,
        n_selected,
        precision,
    )


def assign_hitl(
    score,
    accept_threshold,
    check_threshold,
):
    hitl = np.full(
        len(
            score
        ),
        "Review",
        dtype=object,
    )

    check_mask = (
        score
        >= check_threshold
    )

    accept_mask = (
        score
        >= accept_threshold
    )

    hitl[
        check_mask
    ] = "Check"

    hitl[
        accept_mask
    ] = "Accept"

    return hitl


def category_summary(
    frame,
):
    rows = []

    total = len(
        frame
    )

    for category in [
        "Accept",
        "Check",
        "Review",
    ]:
        part = frame[
            frame[
                "hitl"
            ]
            == category
        ]

        n = len(
            part
        )

        if n == 0:
            accuracy = np.nan
            mean_confidence = np.nan
        else:
            accuracy = float(
                part[
                    "correct"
                ].mean()
            )

            mean_confidence = float(
                part[
                    "calibrated_confidence"
                ].mean()
            )

        rows.append(
            {
                "category":
                category,

                "n_windows":
                n,

                "coverage":
                n
                / total,

                "accuracy":
                accuracy,

                "error_rate":
                (
                    1.0
                    - accuracy
                    if np.isfinite(
                        accuracy
                    )
                    else np.nan
                ),

                "mean_calibrated_confidence":
                mean_confidence,
            }
        )

    return pd.DataFrame(
        rows
    )


def grouped_summary(
    frame,
    group_column,
):
    rows = []

    for group_name in sorted(
        frame[
            group_column
        ].unique()
    ):
        group = frame[
            frame[
                group_column
            ]
            == group_name
        ]

        for category in [
            "Accept",
            "Check",
            "Review",
        ]:
            part = group[
                group[
                    "hitl"
                ]
                == category
            ]

            n = len(
                part
            )

            rows.append(
                {
                    group_column:
                    group_name,

                    "category":
                    category,

                    "n_windows":
                    n,

                    "coverage_within_group":
                    (
                        n
                        / len(
                            group
                        )
                    ),

                    "accuracy":
                    (
                        float(
                            part[
                                "correct"
                            ].mean()
                        )
                        if n > 0
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(
        rows
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
            "Saved results and evaluation data differ"
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

    outer_calibrated_confidence = saved[
        "calibrated_confidence"
    ].to_numpy(
        dtype=float
    )

    final_hitl = np.full(
        len(
            saved
        ),
        "",
        dtype=object,
    )

    accept_thresholds = np.full(
        len(
            saved
        ),
        np.nan,
        dtype=float,
    )

    check_thresholds = np.full(
        len(
            saved
        ),
        np.nan,
        dtype=float,
    )

    fold_rows = []

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
        subject = str(
            groups[
                test_idx[
                    0
                ]
            ]
        )

        print(
            f"fold={fold:02d} "
            f"test={subject} "
            f"windows={len(test_idx)}"
        )

        inner_pred, inner_raw_confidence = (
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

        calibrators = fit_state_calibrators(
            inner_raw_confidence,
            inner_pred,
            inner_correct,
        )

        inner_calibrated_confidence = (
            apply_state_calibrators(
                inner_raw_confidence,
                inner_pred,
                calibrators,
            )
        )

        (
            accept_threshold,
            accept_train_n,
            accept_train_precision,
        ) = threshold_for_target(
            inner_calibrated_confidence,
            inner_correct,
            ACCEPT_TARGET,
        )

        (
            check_threshold,
            non_review_train_n,
            non_review_train_precision,
        ) = threshold_for_target(
            inner_calibrated_confidence,
            inner_correct,
            NON_REVIEW_TARGET,
        )

        if (
            np.isfinite(
                accept_threshold
            )
            and np.isfinite(
                check_threshold
            )
            and check_threshold
            > accept_threshold
        ):
            check_threshold = (
                accept_threshold
            )

        test_score = (
            outer_calibrated_confidence[
                test_idx
            ]
        )

        test_hitl = assign_hitl(
            test_score,
            accept_threshold,
            check_threshold,
        )

        final_hitl[
            test_idx
        ] = test_hitl

        accept_thresholds[
            test_idx
        ] = accept_threshold

        check_thresholds[
            test_idx
        ] = check_threshold

        test_correct = saved[
            "correct"
        ].to_numpy(
            dtype=int
        )[
            test_idx
        ]

        test_accept = (
            test_hitl
            == "Accept"
        )

        test_non_review = (
            test_hitl
            != "Review"
        )

        fold_rows.append(
            {
                "fold":
                fold,

                "subject":
                subject,

                "n_test":
                len(
                    test_idx
                ),

                "accept_threshold":
                accept_threshold,

                "check_threshold":
                check_threshold,

                "train_accept_n":
                accept_train_n,

                "train_accept_coverage":
                accept_train_n
                / len(
                    train_idx
                ),

                "train_accept_precision":
                accept_train_precision,

                "train_non_review_n":
                non_review_train_n,

                "train_non_review_coverage":
                non_review_train_n
                / len(
                    train_idx
                ),

                "train_non_review_precision":
                non_review_train_precision,

                "test_accept_n":
                int(
                    test_accept.sum()
                ),

                "test_accept_accuracy":
                (
                    float(
                        np.mean(
                            test_correct[
                                test_accept
                            ]
                        )
                    )
                    if np.any(
                        test_accept
                    )
                    else np.nan
                ),

                "test_non_review_n":
                int(
                    test_non_review.sum()
                ),

                "test_non_review_accuracy":
                (
                    float(
                        np.mean(
                            test_correct[
                                test_non_review
                            ]
                        )
                    )
                    if np.any(
                        test_non_review
                    )
                    else np.nan
                ),
            }
        )

    if np.any(
        final_hitl
        == ""
    ):
        raise RuntimeError(
            "Missing HITL category"
        )

    output = saved.copy()

    output[
        "hitl"
    ] = final_hitl

    output[
        "accept_threshold"
    ] = accept_thresholds

    output[
        "check_threshold"
    ] = check_thresholds

    print("")
    print("=" * 100)
    print(
        "NESTED HITL — GLOBAL"
    )
    print("=" * 100)

    global_summary = category_summary(
        output
    )

    print(
        global_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    accept_check = output[
        output[
            "hitl"
        ]
        != "Review"
    ]

    review = output[
        output[
            "hitl"
        ]
        == "Review"
    ]

    total_errors = int(
        (
            output[
                "correct"
            ]
            == 0
        ).sum()
    )

    review_errors = int(
        (
            review[
                "correct"
            ]
            == 0
        ).sum()
    )

    print("")
    print(
        f"accept_plus_check_n="
        f"{len(accept_check)}"
    )

    print(
        f"accept_plus_check_coverage="
        f"{len(accept_check) / len(output):.4f}"
    )

    print(
        f"accept_plus_check_accuracy="
        f"{accept_check['correct'].mean():.4f}"
    )

    print(
        f"review_error_capture="
        f"{review_errors / total_errors:.4f}"
    )

    print("")
    print("=" * 100)
    print(
        "NESTED HITL — BY PREDICTED STATE"
    )
    print("=" * 100)

    state_summary = grouped_summary(
        output,
        "predicted_state_name",
    )

    print(
        state_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    print("")
    print("=" * 100)
    print(
        "NESTED HITL — BY SPORT"
    )
    print("=" * 100)

    sport_summary = grouped_summary(
        output,
        "sport",
    )

    print(
        sport_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    fold_summary = pd.DataFrame(
        fold_rows
    )

    print("")
    print("=" * 100)
    print(
        "THRESHOLD DISTRIBUTION"
    )
    print("=" * 100)

    for column in [
        "accept_threshold",
        "check_threshold",
    ]:
        values = fold_summary[
            column
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()

        print(
            f"{column}: "
            f"min={values.min():.4f} "
            f"q25={values.quantile(0.25):.4f} "
            f"median={values.median():.4f} "
            f"q75={values.quantile(0.75):.4f} "
            f"max={values.max():.4f}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_DIR
        / "hitl_nested_predictions.csv",
        index=False,
    )

    global_summary.to_csv(
        OUTPUT_DIR
        / "hitl_nested_global.csv",
        index=False,
    )

    state_summary.to_csv(
        OUTPUT_DIR
        / "hitl_nested_by_state.csv",
        index=False,
    )

    sport_summary.to_csv(
        OUTPUT_DIR
        / "hitl_nested_by_sport.csv",
        index=False,
    )

    fold_summary.to_csv(
        OUTPUT_DIR
        / "hitl_nested_folds.csv",
        index=False,
    )

    print("")
    print(
        "saved="
        "results/experiments/hitl/"
        "hitl_nested_predictions.csv"
    )


if __name__ == "__main__":
    main()
