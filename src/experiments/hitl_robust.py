from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut

from src.core.calibration import (
    inner_oof_predictions,
)

from src.experiments.hitl_nested import (
    apply_state_calibrators,
    assign_hitl,
    category_summary,
    fit_state_calibrators,
    grouped_summary,
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
ACCEPT_Q25_TARGET = 0.90

NON_REVIEW_TARGET = 0.85
NON_REVIEW_Q25_TARGET = 0.75

MIN_SELECTED_PER_SUBJECT = 5
MIN_SUBJECT_FRACTION = 0.50

MIN_SPORT_SUBJECTS = 3
MIN_SPORT_WINDOWS = 300


def robust_threshold(
    score,
    correct,
    subjects,
    target,
    q25_target,
):
    score = np.asarray(
        score,
        dtype=float,
    )

    correct = np.asarray(
        correct,
        dtype=int,
    )

    subjects = np.asarray(
        subjects,
        dtype=str,
    )

    unique_subjects = np.unique(
        subjects
    )

    minimum_subjects = max(
        1,
        int(
            np.ceil(
                len(
                    unique_subjects
                )
                * MIN_SUBJECT_FRACTION
            )
        ),
    )

    candidates = np.unique(
        score
    )

    best = None

    for threshold in candidates:
        selected = (
            score
            >= threshold
        )

        n_selected = int(
            selected.sum()
        )

        if n_selected == 0:
            continue

        pooled_accuracy = float(
            np.mean(
                correct[
                    selected
                ]
            )
        )

        subject_accuracies = []

        for subject in unique_subjects:
            local = (
                selected
                & (
                    subjects
                    == subject
                )
            )

            if int(
                local.sum()
            ) < MIN_SELECTED_PER_SUBJECT:
                continue

            subject_accuracies.append(
                float(
                    np.mean(
                        correct[
                            local
                        ]
                    )
                )
            )

        if len(
            subject_accuracies
        ) < minimum_subjects:
            continue

        subject_accuracies = np.asarray(
            subject_accuracies,
            dtype=float,
        )

        macro_accuracy = float(
            np.mean(
                subject_accuracies
            )
        )

        q25_accuracy = float(
            np.quantile(
                subject_accuracies,
                0.25,
            )
        )

        if (
            pooled_accuracy
            >= target
            and macro_accuracy
            >= target
            and q25_accuracy
            >= q25_target
        ):
            candidate = {
                "threshold":
                float(
                    threshold
                ),

                "n_selected":
                n_selected,

                "coverage":
                n_selected
                / len(
                    score
                ),

                "pooled_accuracy":
                pooled_accuracy,

                "macro_subject_accuracy":
                macro_accuracy,

                "q25_subject_accuracy":
                q25_accuracy,

                "subjects_represented":
                len(
                    subject_accuracies
                ),
            }

            if (
                best is None
                or candidate[
                    "n_selected"
                ]
                > best[
                    "n_selected"
                ]
            ):
                best = candidate

    if best is None:
        return {
            "threshold":
            np.inf,

            "n_selected":
            0,

            "coverage":
            0.0,

            "pooled_accuracy":
            np.nan,

            "macro_subject_accuracy":
            np.nan,

            "q25_subject_accuracy":
            np.nan,

            "subjects_represented":
            0,
        }

    return best


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

    sports = evaluation[
        "sport"
    ].to_numpy(
        dtype=str
    )

    outer_confidence = saved[
        "calibrated_confidence"
    ].to_numpy(
        dtype=float
    )

    correct = saved[
        "correct"
    ].to_numpy(
        dtype=int
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

    threshold_sources = np.full(
        len(
            saved
        ),
        "",
        dtype=object,
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

        sport = str(
            sports[
                test_idx[
                    0
                ]
            ]
        )

        print(
            f"fold={fold:02d} "
            f"test={subject} "
            f"sport={sport} "
            f"windows={len(test_idx)}"
        )

        (
            inner_prediction,
            inner_raw_confidence,
        ) = inner_oof_predictions(
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

        inner_correct = (
            inner_prediction
            == y[
                train_idx
            ]
        ).astype(
            int
        )

        calibrators = fit_state_calibrators(
            inner_raw_confidence,
            inner_prediction,
            inner_correct,
        )

        inner_confidence = apply_state_calibrators(
            inner_raw_confidence,
            inner_prediction,
            calibrators,
        )

        inner_subjects = groups[
            train_idx
        ]

        inner_sports = sports[
            train_idx
        ]

        sport_mask = (
            inner_sports
            == sport
        )

        sport_subjects = np.unique(
            inner_subjects[
                sport_mask
            ]
        )

        if (
            int(
                sport_mask.sum()
            )
            >= MIN_SPORT_WINDOWS
            and len(
                sport_subjects
            )
            >= MIN_SPORT_SUBJECTS
        ):
            threshold_mask = (
                sport_mask
            )

            threshold_source = (
                "sport"
            )
        else:
            threshold_mask = np.ones(
                len(
                    inner_correct
                ),
                dtype=bool,
            )

            threshold_source = (
                "global_fallback"
            )

        accept = robust_threshold(
            inner_confidence[
                threshold_mask
            ],
            inner_correct[
                threshold_mask
            ],
            inner_subjects[
                threshold_mask
            ],
            ACCEPT_TARGET,
            ACCEPT_Q25_TARGET,
        )

        non_review = robust_threshold(
            inner_confidence[
                threshold_mask
            ],
            inner_correct[
                threshold_mask
            ],
            inner_subjects[
                threshold_mask
            ],
            NON_REVIEW_TARGET,
            NON_REVIEW_Q25_TARGET,
        )

        accept_threshold = accept[
            "threshold"
        ]

        check_threshold = non_review[
            "threshold"
        ]

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

        test_score = outer_confidence[
            test_idx
        ]

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

        threshold_sources[
            test_idx
        ] = threshold_source

        test_correct = correct[
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

                "sport":
                sport,

                "threshold_source":
                threshold_source,

                "n_test":
                len(
                    test_idx
                ),

                "accept_threshold":
                accept_threshold,

                "check_threshold":
                check_threshold,

                "train_accept_coverage":
                accept[
                    "coverage"
                ],

                "train_accept_accuracy":
                accept[
                    "pooled_accuracy"
                ],

                "train_accept_macro_subject":
                accept[
                    "macro_subject_accuracy"
                ],

                "train_accept_q25_subject":
                accept[
                    "q25_subject_accuracy"
                ],

                "train_accept_subjects":
                accept[
                    "subjects_represented"
                ],

                "train_non_review_coverage":
                non_review[
                    "coverage"
                ],

                "train_non_review_accuracy":
                non_review[
                    "pooled_accuracy"
                ],

                "train_non_review_macro_subject":
                non_review[
                    "macro_subject_accuracy"
                ],

                "train_non_review_q25_subject":
                non_review[
                    "q25_subject_accuracy"
                ],

                "train_non_review_subjects":
                non_review[
                    "subjects_represented"
                ],

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
        "hitl_robust"
    ] = final_hitl

    output[
        "accept_threshold_robust"
    ] = accept_thresholds

    output[
        "check_threshold_robust"
    ] = check_thresholds

    output[
        "threshold_source"
    ] = threshold_sources

    analysis = output.rename(
        columns={
            "hitl_robust":
            "hitl",
        }
    )

    print("")
    print("=" * 110)
    print(
        "ROBUST NESTED HITL — GLOBAL"
    )
    print("=" * 110)

    global_summary = category_summary(
        analysis
    )

    print(
        global_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    non_review = analysis[
        analysis[
            "hitl"
        ]
        != "Review"
    ]

    review = analysis[
        analysis[
            "hitl"
        ]
        == "Review"
    ]

    total_errors = int(
        (
            analysis[
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
        f"{len(non_review)}"
    )

    print(
        f"accept_plus_check_coverage="
        f"{len(non_review) / len(analysis):.4f}"
    )

    print(
        f"accept_plus_check_accuracy="
        f"{non_review['correct'].mean():.4f}"
    )

    print(
        f"review_error_capture="
        f"{review_errors / total_errors:.4f}"
    )

    print("")
    print("=" * 110)
    print(
        "ROBUST HITL — BY PREDICTED STATE"
    )
    print("=" * 110)

    state_summary = grouped_summary(
        analysis,
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
    print("=" * 110)
    print(
        "ROBUST HITL — BY SPORT"
    )
    print("=" * 110)

    sport_summary = grouped_summary(
        analysis,
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
    print("=" * 110)
    print(
        "OUTER SUBJECT PERFORMANCE"
    )
    print("=" * 110)

    valid_accept = fold_summary[
        fold_summary[
            "test_accept_n"
        ]
        > 0
    ]

    print(
        f"subjects_with_accept="
        f"{len(valid_accept)}/"
        f"{len(fold_summary)}"
    )

    if len(
        valid_accept
    ) > 0:
        values = valid_accept[
            "test_accept_accuracy"
        ]

        print(
            f"accept_accuracy: "
            f"min={values.min():.4f} "
            f"q25={values.quantile(0.25):.4f} "
            f"median={values.median():.4f} "
            f"q75={values.quantile(0.75):.4f} "
            f"max={values.max():.4f}"
        )

    print("")
    print(
        fold_summary[
            [
                "subject",
                "sport",
                "threshold_source",
                "test_accept_n",
                "test_accept_accuracy",
                "test_non_review_n",
                "test_non_review_accuracy",
                "accept_threshold",
                "check_threshold",
            ]
        ]
        .sort_values(
            "test_accept_accuracy",
            na_position="last",
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_DIR
        / "hitl_robust_predictions.csv",
        index=False,
    )

    global_summary.to_csv(
        OUTPUT_DIR
        / "hitl_robust_global.csv",
        index=False,
    )

    state_summary.to_csv(
        OUTPUT_DIR
        / "hitl_robust_by_state.csv",
        index=False,
    )

    sport_summary.to_csv(
        OUTPUT_DIR
        / "hitl_robust_by_sport.csv",
        index=False,
    )

    fold_summary.to_csv(
        OUTPUT_DIR
        / "hitl_robust_folds.csv",
        index=False,
    )

    print("")
    print(
        "saved="
        "results/experiments/hitl/"
        "hitl_robust_predictions.csv"
    )


if __name__ == "__main__":
    main()
