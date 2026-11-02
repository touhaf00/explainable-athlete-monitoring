from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)


INPUT = Path(
    "results/baseline/xai_oof_predictions_calibrated.csv"
)

FOLDS_INPUT = Path(
    "results/experiments/hitl/hitl_nested_folds.csv"
)

N_BINS = 10


def ece(
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

    result = 0.0

    for index in range(
        N_BINS
    ):
        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if index == N_BINS - 1:
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

        result += (
            n
            / total
        ) * abs(
            np.mean(
                correct[
                    mask
                ]
            )
            - np.mean(
                score[
                    mask
                ]
            )
        )

    return float(
        result
    )


def metrics(
    frame,
):
    correct = frame[
        "correct"
    ].to_numpy(
        dtype=int
    )

    score = frame[
        "calibrated_confidence"
    ].to_numpy(
        dtype=float
    )

    if len(
        np.unique(
            correct
        )
    ) == 2:
        auc = roc_auc_score(
            correct,
            score,
        )
    else:
        auc = np.nan

    accuracy = float(
        np.mean(
            correct
        )
    )

    mean_score = float(
        np.mean(
            score
        )
    )

    return {
        "n_windows":
        len(
            frame
        ),

        "accuracy":
        accuracy,

        "mean_confidence":
        mean_score,

        "bias":
        accuracy
        - mean_score,

        "brier":
        brier_score_loss(
            correct,
            score,
        ),

        "ece":
        ece(
            correct,
            score,
        ),

        "roc_auc":
        auc,
    }


def main():
    frame = pd.read_csv(
        INPUT
    )

    print(
        f"windows={len(frame)}"
    )

    print("")
    print("=" * 110)
    print(
        "CALIBRATED CONFIDENCE BY SPORT"
    )
    print("=" * 110)

    rows = []

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

        rows.append(
            {
                "sport":
                sport,

                **metrics(
                    part
                ),
            }
        )

    sport_frame = pd.DataFrame(
        rows
    )

    print(
        sport_frame.round(
            4
        ).to_string(
            index=False
        )
    )

    print("")
    print("=" * 110)
    print(
        "CALIBRATED CONFIDENCE BY PREDICTED STATE"
    )
    print("=" * 110)

    rows = []

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

        rows.append(
            {
                "predicted_state":
                state,

                **metrics(
                    part
                ),
            }
        )

    state_frame = pd.DataFrame(
        rows
    )

    print(
        state_frame.round(
            4
        ).to_string(
            index=False
        )
    )

    print("")
    print("=" * 110)
    print(
        "CALIBRATED CONFIDENCE BY SPORT × PREDICTED STATE"
    )
    print("=" * 110)

    rows = []

    for sport in sorted(
        frame[
            "sport"
        ].unique()
    ):
        for state in [
            "Rest",
            "Effort",
            "Recovery",
        ]:
            part = frame[
                (
                    frame[
                        "sport"
                    ]
                    == sport
                )
                & (
                    frame[
                        "predicted_state_name"
                    ]
                    == state
                )
            ]

            if len(
                part
            ) == 0:
                continue

            rows.append(
                {
                    "sport":
                    sport,

                    "predicted_state":
                    state,

                    **metrics(
                        part
                    ),
                }
            )

    context_frame = pd.DataFrame(
        rows
    )

    print(
        context_frame.round(
            4
        ).to_string(
            index=False
        )
    )

    folds = pd.read_csv(
        FOLDS_INPUT
    )

    print("")
    print("=" * 110)
    print(
        "OUTER SUBJECT HITL DISTRIBUTION"
    )
    print("=" * 110)

    for column in [
        "test_accept_accuracy",
        "test_non_review_accuracy",
    ]:
        values = folds[
            column
        ].dropna()

        print(
            f"{column}: "
            f"min={values.min():.4f} "
            f"q25={values.quantile(0.25):.4f} "
            f"median={values.median():.4f} "
            f"q75={values.quantile(0.75):.4f} "
            f"max={values.max():.4f}"
        )

    subjects_with_accept = int(
        folds[
            "test_accept_n"
        ].gt(
            0
        ).sum()
    )

    print(
        f"subjects_with_accept="
        f"{subjects_with_accept}/"
        f"{len(folds)}"
    )

    print("")
    print("=" * 110)
    print(
        "WORST SUBJECTS WITH ACCEPT"
    )
    print("=" * 110)

    worst = (
        folds[
            folds[
                "test_accept_n"
            ]
            > 0
        ]
        .sort_values(
            "test_accept_accuracy",
            ascending=True,
        )
        [
            [
                "subject",
                "n_test",
                "test_accept_n",
                "test_accept_accuracy",
                "test_non_review_n",
                "test_non_review_accuracy",
                "accept_threshold",
                "check_threshold",
            ]
        ]
    )

    print(
        worst.round(
            4
        ).to_string(
            index=False
        )
    )

    sport_frame.to_csv(
        "results/experiments/context_calibration/"
        "calibrated_confidence_by_sport.csv",
        index=False,
    )

    state_frame.to_csv(
        "results/experiments/context_calibration/"
        "calibrated_confidence_by_state.csv",
        index=False,
    )

    context_frame.to_csv(
        "results/experiments/context_calibration/"
        "calibrated_confidence_by_context.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
