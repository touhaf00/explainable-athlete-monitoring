from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


INPUT = Path(
    "results/baseline/xai_oof_predictions_calibrated.csv"
)

OUTPUT = Path(
    "results/experiments/semantic_normalization/xai_oof_predictions_semantic_normalized.csv"
)

MIN_GROUP_WINDOWS = 100
MIN_GROUP_SUBJECTS = 2

COVERAGES = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
]


def percentile_from_training(
    training_values,
    test_values,
):
    training_values = np.sort(
        np.asarray(
            training_values,
            dtype=float,
        )
    )

    test_values = np.asarray(
        test_values,
        dtype=float,
    )

    ranks = np.searchsorted(
        training_values,
        test_values,
        side="right",
    )

    return (
        ranks
        + 0.5
    ) / (
        len(
            training_values
        )
        + 1.0
    )


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


def score_summary(
    frame,
    score_name,
):
    correct = frame[
        "correct"
    ].to_numpy(
        dtype=int
    )

    score = frame[
        score_name
    ].to_numpy(
        dtype=float
    )

    row = {
        "score":
        score_name,

        "roc_auc":
        roc_auc_score(
            correct,
            score,
        ),
    }

    for coverage in COVERAGES:
        row[
            f"coverage_{int(coverage * 100)}"
        ] = selective_accuracy(
            correct,
            score,
            coverage,
        )

    return row


def main():
    frame = pd.read_csv(
        INPUT
    )

    n = len(
        frame
    )

    state_percentile = np.full(
        n,
        np.nan,
        dtype=float,
    )

    state_sport_percentile = np.full(
        n,
        np.nan,
        dtype=float,
    )

    subjects = frame[
        "subject"
    ].unique()

    for fold, subject in enumerate(
        subjects,
        start=1,
    ):
        train_mask = (
            frame[
                "subject"
            ]
            != subject
        )

        test_mask = (
            frame[
                "subject"
            ]
            == subject
        )

        train = frame[
            train_mask
        ]

        test = frame[
            test_mask
        ]

        print(
            f"fold={fold:02d} "
            f"test={subject} "
            f"windows={len(test)}"
        )

        for state in test[
            "predicted_state_name"
        ].unique():
            state_train = train[
                train[
                    "predicted_state_name"
                ]
                == state
            ]

            state_test_mask = (
                test[
                    "predicted_state_name"
                ]
                == state
            )

            state_test = test[
                state_test_mask
            ]

            state_values = percentile_from_training(
                state_train[
                    "consistency"
                ].to_numpy(),
                state_test[
                    "consistency"
                ].to_numpy(),
            )

            state_percentile[
                state_test.index
            ] = state_values

            for sport in state_test[
                "sport"
            ].unique():
                local_test_mask = (
                    (
                        state_test[
                            "sport"
                        ]
                        == sport
                    )
                )

                local_test = state_test[
                    local_test_mask
                ]

                state_sport_train = state_train[
                    state_train[
                        "sport"
                    ]
                    == sport
                ]

                enough_windows = (
                    len(
                        state_sport_train
                    )
                    >= MIN_GROUP_WINDOWS
                )

                enough_subjects = (
                    state_sport_train[
                        "subject"
                    ].nunique()
                    >= MIN_GROUP_SUBJECTS
                )

                if (
                    enough_windows
                    and enough_subjects
                ):
                    reference = (
                        state_sport_train[
                            "consistency"
                        ].to_numpy()
                    )
                else:
                    reference = (
                        state_train[
                            "consistency"
                        ].to_numpy()
                    )

                local_values = percentile_from_training(
                    reference,
                    local_test[
                        "consistency"
                    ].to_numpy(),
                )

                state_sport_percentile[
                    local_test.index
                ] = local_values

    if np.any(
        ~np.isfinite(
            state_percentile
        )
    ):
        raise RuntimeError(
            "Missing state consistency percentile"
        )

    if np.any(
        ~np.isfinite(
            state_sport_percentile
        )
    ):
        raise RuntimeError(
            "Missing state-sport consistency percentile"
        )

    frame[
        "consistency_state_percentile"
    ] = state_percentile

    frame[
        "consistency_state_sport_percentile"
    ] = state_sport_percentile

    frame[
        "reliability_calibrated_state"
    ] = np.sqrt(
        frame[
            "calibrated_confidence"
        ]
        * frame[
            "consistency_state_percentile"
        ]
    )

    frame[
        "reliability_calibrated_state_sport"
    ] = np.sqrt(
        frame[
            "calibrated_confidence"
        ]
        * frame[
            "consistency_state_sport_percentile"
        ]
    )

    scores = [
        "calibrated_confidence",
        "consistency",
        "consistency_state_percentile",
        "consistency_state_sport_percentile",
        "calibrated_reliability",
        "reliability_calibrated_state",
        "reliability_calibrated_state_sport",
    ]

    rows = [
        score_summary(
            frame,
            score,
        )
        for score in scores
    ]

    summary = pd.DataFrame(
        rows
    )

    print("")
    print("=" * 110)
    print(
        "GLOBAL SCORE QUALITY"
    )
    print("=" * 110)

    print(
        summary.round(
            4
        ).to_string(
            index=False
        )
    )

    state_rows = []

    for state in [
        "Rest",
        "Effort",
        "Recovery",
    ]:
        part = frame[
            frame[
                "predicted_state_name"
            ]
            == state
        ]

        for score in [
            "calibrated_confidence",
            "consistency_state_percentile",
            "consistency_state_sport_percentile",
            "reliability_calibrated_state",
            "reliability_calibrated_state_sport",
        ]:
            state_rows.append(
                {
                    "state":
                    state,

                    "n_windows":
                    len(
                        part
                    ),

                    **score_summary(
                        part,
                        score,
                    ),
                }
            )

    state_summary = pd.DataFrame(
        state_rows
    )

    print("")
    print("=" * 110)
    print(
        "QUALITY BY PREDICTED STATE"
    )
    print("=" * 110)

    print(
        state_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    sport_rows = []

    for sport in sorted(
        frame[
            "sport"
        ].unique()
    ):
        part = frame[
            frame[
                "sport"
            ]
            == sport
        ]

        for score in [
            "calibrated_confidence",
            "consistency_state_percentile",
            "consistency_state_sport_percentile",
            "reliability_calibrated_state",
            "reliability_calibrated_state_sport",
        ]:
            sport_rows.append(
                {
                    "sport":
                    sport,

                    "n_windows":
                    len(
                        part
                    ),

                    **score_summary(
                        part,
                        score,
                    ),
                }
            )

    sport_summary = pd.DataFrame(
        sport_rows
    )

    print("")
    print("=" * 110)
    print(
        "QUALITY BY SPORT"
    )
    print("=" * 110)

    print(
        sport_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        OUTPUT,
        index=False,
    )

    summary.to_csv(
        OUTPUT.parent
        / "semantic_normalization_global.csv",
        index=False,
    )

    state_summary.to_csv(
        OUTPUT.parent
        / "semantic_normalization_by_state.csv",
        index=False,
    )

    sport_summary.to_csv(
        OUTPUT.parent
        / "semantic_normalization_by_sport.csv",
        index=False,
    )

    print("")
    print(
        "saved="
        "results/experiments/semantic_normalization/"
        "xai_oof_predictions_semantic_normalized.csv"
    )


if __name__ == "__main__":
    main()
