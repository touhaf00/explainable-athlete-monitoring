from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut

from src.core.calibration import (
    apply_calibrator,
    calibration_metrics,
    fit_calibrator,
    inner_oof_predictions,
    selective_accuracy,
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

OUTPUT = Path(
    "results/experiments/context_calibration/xai_oof_predictions_context_calibrated.csv"
)

FEATURE_SET = "personalized_14"

MIN_STATE_WINDOWS = 100
MIN_CONTEXT_WINDOWS = 100
MIN_CONTEXT_SUBJECTS = 3


def fit_calibrators(
    confidence,
    prediction,
    correct,
    sports,
    subjects,
):
    global_calibrator = fit_calibrator(
        confidence,
        correct,
    )

    if global_calibrator is None:
        raise RuntimeError(
            "Global calibrator could not be fitted"
        )

    state_calibrators = {}

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
            >= MIN_STATE_WINDOWS
            and len(
                np.unique(
                    state_correct
                )
            )
            == 2
        ):
            state_calibrators[
                state
            ] = fit_calibrator(
                confidence[
                    mask
                ],
                state_correct,
            )
        else:
            state_calibrators[
                state
            ] = global_calibrator

    context_calibrators = {}

    for sport in np.unique(
        sports
    ):
        for state in [
            0,
            1,
            2,
        ]:
            mask = (
                (sports == sport)
                & (
                    prediction
                    == state
                )
            )

            n_windows = int(
                mask.sum()
            )

            n_subjects = len(
                np.unique(
                    subjects[
                        mask
                    ]
                )
            )

            context_correct = correct[
                mask
            ]

            if (
                n_windows
                >= MIN_CONTEXT_WINDOWS
                and n_subjects
                >= MIN_CONTEXT_SUBJECTS
                and len(
                    np.unique(
                        context_correct
                    )
                )
                == 2
            ):
                context_calibrators[
                    (
                        sport,
                        state,
                    )
                ] = fit_calibrator(
                    confidence[
                        mask
                    ],
                    context_correct,
                )

    return (
        global_calibrator,
        state_calibrators,
        context_calibrators,
    )


def apply_context_calibration(
    confidence,
    prediction,
    sports,
    state_calibrators,
    context_calibrators,
):
    calibrated = np.full(
        len(
            confidence
        ),
        np.nan,
        dtype=float,
    )

    source = np.full(
        len(
            confidence
        ),
        "",
        dtype=object,
    )

    for index in range(
        len(
            confidence
        )
    ):
        state = int(
            prediction[
                index
            ]
        )

        sport = sports[
            index
        ]

        key = (
            sport,
            state,
        )

        if key in context_calibrators:
            model = context_calibrators[
                key
            ]

            source[
                index
            ] = "sport_state"
        else:
            model = state_calibrators[
                state
            ]

            source[
                index
            ] = "state_fallback"

        calibrated[
            index
        ] = apply_calibrator(
            model,
            np.asarray(
                [
                    confidence[
                        index
                    ]
                ]
            ),
        )[
            0
        ]

    return (
        calibrated,
        source,
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

    context_confidence = np.full(
        len(
            saved
        ),
        np.nan,
        dtype=float,
    )

    calibration_source = np.full(
        len(
            saved
        ),
        "",
        dtype=object,
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

        (
            inner_prediction,
            inner_confidence,
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

        (
            global_calibrator,
            state_calibrators,
            context_calibrators,
        ) = fit_calibrators(
            inner_confidence,
            inner_prediction,
            inner_correct,
            sports[
                train_idx
            ],
            groups[
                train_idx
            ],
        )

        (
            fold_confidence,
            fold_source,
        ) = apply_context_calibration(
            raw_confidence[
                test_idx
            ],
            outer_prediction[
                test_idx
            ],
            sports[
                test_idx
            ],
            state_calibrators,
            context_calibrators,
        )

        context_confidence[
            test_idx
        ] = fold_confidence

        calibration_source[
            test_idx
        ] = fold_source

    if np.any(
        ~np.isfinite(
            context_confidence
        )
    ):
        raise RuntimeError(
            "Missing context calibrated confidence"
        )

    state_confidence = saved[
        "calibrated_confidence"
    ].to_numpy(
        dtype=float
    )

    print("")
    print("=" * 110)
    print(
        "GLOBAL CALIBRATION"
    )
    print("=" * 110)

    global_rows = []

    for name, score in [
        (
            "state_calibrated",
            state_confidence,
        ),
        (
            "context_calibrated",
            context_confidence,
        ),
    ]:
        metrics = calibration_metrics(
            correct,
            score,
        )

        global_rows.append(
            {
                "score":
                name,

                **metrics,
            }
        )

    print(
        pd.DataFrame(
            global_rows
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("")
    print("=" * 110)
    print(
        "SELECTIVE ACCURACY"
    )
    print("=" * 110)

    selective_rows = []

    for name, score in [
        (
            "state_calibrated",
            state_confidence,
        ),
        (
            "context_calibrated",
            context_confidence,
        ),
    ]:
        row = {
            "score":
            name,
        }

        for coverage in [
            0.20,
            0.40,
            0.60,
            0.80,
        ]:
            row[
                f"coverage_{int(coverage * 100)}"
            ] = selective_accuracy(
                correct,
                score,
                coverage,
            )

        selective_rows.append(
            row
        )

    print(
        pd.DataFrame(
            selective_rows
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("")
    print("=" * 110)
    print(
        "CALIBRATION BY SPORT × PREDICTED STATE"
    )
    print("=" * 110)

    rows = []

    for sport in sorted(
        np.unique(
            sports
        )
    ):
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
                (sports == sport)
                & (
                    outer_prediction
                    == state
                )
            )

            if not np.any(
                mask
            ):
                continue

            for name, score in [
                (
                    "state",
                    state_confidence,
                ),
                (
                    "context",
                    context_confidence,
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

                rows.append(
                    {
                        "sport":
                        sport,

                        "predicted_state":
                        state_name,

                        "score":
                        name,

                        **metrics,
                    }
                )

    context_summary = pd.DataFrame(
        rows
    )

    print(
        context_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    print("")
    print("=" * 110)
    print(
        "CALIBRATION SOURCE"
    )
    print("=" * 110)

    source_counts = (
        pd.Series(
            calibration_source
        )
        .value_counts()
        .rename_axis(
            "source"
        )
        .reset_index(
            name="n_windows"
        )
    )

    source_counts[
        "coverage"
    ] = (
        source_counts[
            "n_windows"
        ]
        / len(
            saved
        )
    )

    print(
        source_counts.round(
            4
        ).to_string(
            index=False
        )
    )

    output = saved.copy()

    output[
        "context_calibrated_confidence"
    ] = context_confidence

    output[
        "context_calibration_source"
    ] = calibration_source

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    context_summary.to_csv(
        OUTPUT.parent
        / "context_calibration_summary.csv",
        index=False,
    )

    pd.DataFrame(
        global_rows
    ).to_csv(
        OUTPUT.parent
        / "context_calibration_global.csv",
        index=False,
    )

    print("")
    print(
        "saved="
        "results/experiments/context_calibration/"
        "xai_oof_predictions_context_calibrated.csv"
    )


if __name__ == "__main__":
    main()
