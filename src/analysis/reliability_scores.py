from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


INPUT = Path(
    "results/baseline/xai_oof_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/experiments/reliability"
)

SCORES = [
    "confidence",
    "consistency",
    "reliability",
]

COVERAGES = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
]

BOOTSTRAPS = 1000
RANDOM_SEED = 42


def score_metrics(
    frame,
    score,
):
    y = frame[
        "correct"
    ].to_numpy(
        dtype=int
    )

    values = frame[
        score
    ].to_numpy(
        dtype=float
    )

    if len(
        np.unique(
            y
        )
    ) < 2:
        return {
            "roc_auc": np.nan,
            "average_precision": np.nan,
        }

    return {
        "roc_auc":
        roc_auc_score(
            y,
            values,
        ),

        "average_precision":
        average_precision_score(
            y,
            values,
        ),
    }


def selective_accuracy(
    frame,
    score,
):
    ordered = (
        frame.sort_values(
            score,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    rows = []

    for coverage in COVERAGES:
        n = int(
            np.ceil(
                len(
                    ordered
                )
                * coverage
            )
        )

        selected = ordered.iloc[
            :n
        ]

        accuracy = float(
            selected[
                "correct"
            ].mean()
        )

        rows.append(
            {
                "score": score,
                "coverage": coverage,
                "n_windows": n,
                "accuracy": accuracy,
                "error_rate": 1.0 - accuracy,
            }
        )

    return rows


def bootstrap_auc(
    frame,
):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    subjects = frame[
        "subject"
    ].unique()

    subject_indices = {
        subject:
        np.where(
            frame[
                "subject"
            ].to_numpy()
            == subject
        )[0]
        for subject in subjects
    }

    rows = []

    for iteration in range(
        BOOTSTRAPS
    ):
        sampled_subjects = rng.choice(
            subjects,
            size=len(
                subjects
            ),
            replace=True,
        )

        indices = np.concatenate(
            [
                subject_indices[
                    subject
                ]
                for subject
                in sampled_subjects
            ]
        )

        sample = frame.iloc[
            indices
        ]

        y = sample[
            "correct"
        ].to_numpy(
            dtype=int
        )

        if len(
            np.unique(
                y
            )
        ) < 2:
            continue

        auc_confidence = (
            roc_auc_score(
                y,
                sample[
                    "confidence"
                ],
            )
        )

        auc_consistency = (
            roc_auc_score(
                y,
                sample[
                    "consistency"
                ],
            )
        )

        auc_reliability = (
            roc_auc_score(
                y,
                sample[
                    "reliability"
                ],
            )
        )

        rows.append(
            {
                "iteration":
                iteration,

                "auc_confidence":
                auc_confidence,

                "auc_consistency":
                auc_consistency,

                "auc_reliability":
                auc_reliability,

                "reliability_minus_confidence":
                auc_reliability
                - auc_confidence,

                "consistency_minus_confidence":
                auc_consistency
                - auc_confidence,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    frame = pd.read_csv(
        INPUT
    )

    print(
        f"windows={len(frame)}"
    )

    print(
        f"subjects="
        f"{frame['subject'].nunique()}"
    )

    print(
        f"overall_accuracy="
        f"{frame['correct'].mean():.4f}"
    )

    summary_rows = []

    for score in SCORES:
        metrics = score_metrics(
            frame,
            score,
        )

        summary_rows.append(
            {
                "score":
                score,

                **metrics,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print("")
    print("=" * 90)
    print(
        "GLOBAL SCORE QUALITY"
    )
    print("=" * 90)

    print(
        summary.round(
            4
        ).to_string(
            index=False
        )
    )

    selective_rows = []

    for score in SCORES:
        selective_rows.extend(
            selective_accuracy(
                frame,
                score,
            )
        )

    selective = pd.DataFrame(
        selective_rows
    )

    print("")
    print("=" * 90)
    print(
        "SELECTIVE ACCURACY"
    )
    print("=" * 90)

    selective_display = (
        selective.pivot(
            index="coverage",
            columns="score",
            values="accuracy",
        )
        .reset_index()
    )

    print(
        selective_display.round(
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

        for score in SCORES:
            metrics = score_metrics(
                part,
                score,
            )

            state_rows.append(
                {
                    "predicted_state":
                    state,

                    "score":
                    score,

                    "n_windows":
                    len(
                        part
                    ),

                    "accuracy":
                    part[
                        "correct"
                    ].mean(),

                    **metrics,
                }
            )

    state_summary = pd.DataFrame(
        state_rows
    )

    print("")
    print("=" * 90)
    print(
        "QUALITY BY PREDICTED STATE"
    )
    print("=" * 90)

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

        for score in SCORES:
            metrics = score_metrics(
                part,
                score,
            )

            sport_rows.append(
                {
                    "sport":
                    sport,

                    "score":
                    score,

                    "n_windows":
                    len(
                        part
                    ),

                    "accuracy":
                    part[
                        "correct"
                    ].mean(),

                    **metrics,
                }
            )

    sport_summary = pd.DataFrame(
        sport_rows
    )

    print("")
    print("=" * 90)
    print(
        "QUALITY BY SPORT"
    )
    print("=" * 90)

    print(
        sport_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    print("")
    print(
        "Running subject-level bootstrap..."
    )

    bootstrap = bootstrap_auc(
        frame
    )

    print("")
    print("=" * 90)
    print(
        "SUBJECT-LEVEL BOOTSTRAP"
    )
    print("=" * 90)

    for column in [
        "reliability_minus_confidence",
        "consistency_minus_confidence",
    ]:
        values = bootstrap[
            column
        ].to_numpy(
            dtype=float
        )

        print(
            f"{column}: "
            f"mean={np.mean(values):.4f} "
            f"ci95=["
            f"{np.quantile(values, 0.025):.4f}, "
            f"{np.quantile(values, 0.975):.4f}"
            f"]"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "score_quality_global.csv",
        index=False,
    )

    selective.to_csv(
        OUTPUT_DIR
        / "score_selective_accuracy.csv",
        index=False,
    )

    state_summary.to_csv(
        OUTPUT_DIR
        / "score_quality_by_state.csv",
        index=False,
    )

    sport_summary.to_csv(
        OUTPUT_DIR
        / "score_quality_by_sport.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUTPUT_DIR
        / "score_bootstrap.csv",
        index=False,
    )

    print("")
    print(
        "saved="
        "results/experiments/reliability/"
        "score_quality_global.csv"
    )

    print(
        "saved="
        "results/experiments/reliability/"
        "score_selective_accuracy.csv"
    )

    print(
        "saved="
        "results/experiments/reliability/"
        "score_quality_by_state.csv"
    )

    print(
        "saved="
        "results/experiments/reliability/"
        "score_quality_by_sport.csv"
    )

    print(
        "saved="
        "results/experiments/reliability/"
        "score_bootstrap.csv"
    )


if __name__ == "__main__":
    main()
