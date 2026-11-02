from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "results/baseline/xai_oof_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/baseline"
)

N_BINS = 10
BOOTSTRAPS = 1000
RANDOM_SEED = 42


def brier_score(
    y,
    confidence,
):
    return float(
        np.mean(
            (
                confidence
                - y
            ) ** 2
        )
    )


def calibration_table(
    y,
    confidence,
    n_bins=N_BINS,
):
    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for index in range(
        n_bins
    ):
        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if index == (
            n_bins - 1
        ):
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        n = int(
            mask.sum()
        )

        if n == 0:
            continue

        mean_confidence = float(
            np.mean(
                confidence[
                    mask
                ]
            )
        )

        empirical_accuracy = float(
            np.mean(
                y[
                    mask
                ]
            )
        )

        gap = (
            empirical_accuracy
            - mean_confidence
        )

        rows.append(
            {
                "bin":
                index + 1,

                "lower":
                lower,

                "upper":
                upper,

                "n_windows":
                n,

                "mean_confidence":
                mean_confidence,

                "empirical_accuracy":
                empirical_accuracy,

                "gap_accuracy_minus_confidence":
                gap,

                "absolute_gap":
                abs(
                    gap
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def adaptive_calibration_table(
    y,
    confidence,
    n_bins=N_BINS,
):
    order = np.argsort(
        confidence
    )

    groups = np.array_split(
        order,
        n_bins,
    )

    rows = []

    for index, indices in enumerate(
        groups,
        start=1,
    ):
        if len(
            indices
        ) == 0:
            continue

        values = confidence[
            indices
        ]

        labels = y[
            indices
        ]

        mean_confidence = float(
            np.mean(
                values
            )
        )

        empirical_accuracy = float(
            np.mean(
                labels
            )
        )

        gap = (
            empirical_accuracy
            - mean_confidence
        )

        rows.append(
            {
                "bin":
                index,

                "lower":
                float(
                    np.min(
                        values
                    )
                ),

                "upper":
                float(
                    np.max(
                        values
                    )
                ),

                "n_windows":
                len(
                    indices
                ),

                "mean_confidence":
                mean_confidence,

                "empirical_accuracy":
                empirical_accuracy,

                "gap_accuracy_minus_confidence":
                gap,

                "absolute_gap":
                abs(
                    gap
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def ece_from_table(
    table,
):
    total = table[
        "n_windows"
    ].sum()

    return float(
        np.sum(
            (
                table[
                    "n_windows"
                ]
                / total
            )
            * table[
                "absolute_gap"
            ]
        )
    )


def calibration_metrics(
    frame,
):
    y = frame[
        "correct"
    ].to_numpy(
        dtype=float
    )

    confidence = frame[
        "confidence"
    ].to_numpy(
        dtype=float
    )

    fixed = calibration_table(
        y,
        confidence,
    )

    adaptive = adaptive_calibration_table(
        y,
        confidence,
    )

    brier = brier_score(
        y,
        confidence,
    )

    prevalence = float(
        np.mean(
            y
        )
    )

    baseline_brier = float(
        np.mean(
            (
                prevalence
                - y
            ) ** 2
        )
    )

    if baseline_brier > 0:
        skill = (
            1.0
            - brier
            / baseline_brier
        )
    else:
        skill = np.nan

    return {
        "n_windows":
        len(
            frame
        ),

        "accuracy":
        prevalence,

        "mean_confidence":
        float(
            np.mean(
                confidence
            )
        ),

        "bias_accuracy_minus_confidence":
        prevalence
        - float(
            np.mean(
                confidence
            )
        ),

        "brier":
        brier,

        "baseline_brier":
        baseline_brier,

        "brier_skill":
        skill,

        "ece_fixed":
        ece_from_table(
            fixed
        ),

        "ece_adaptive":
        ece_from_table(
            adaptive
        ),

        "max_calibration_error":
        float(
            fixed[
                "absolute_gap"
            ].max()
        ),

        "fixed_table":
        fixed,

        "adaptive_table":
        adaptive,
    }


def bootstrap_subjects(
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
        for subject
        in subjects
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

        sample = (
            frame.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        )

        metrics = calibration_metrics(
            sample
        )

        rows.append(
            {
                "iteration":
                iteration,

                "brier":
                metrics[
                    "brier"
                ],

                "brier_skill":
                metrics[
                    "brier_skill"
                ],

                "ece_fixed":
                metrics[
                    "ece_fixed"
                ],

                "ece_adaptive":
                metrics[
                    "ece_adaptive"
                ],

                "bias":
                metrics[
                    "bias_accuracy_minus_confidence"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


def summary_row(
    frame,
    group_type,
    group_name,
):
    metrics = calibration_metrics(
        frame
    )

    return {
        "group_type":
        group_type,

        "group_name":
        group_name,

        "n_windows":
        metrics[
            "n_windows"
        ],

        "accuracy":
        metrics[
            "accuracy"
        ],

        "mean_confidence":
        metrics[
            "mean_confidence"
        ],

        "bias_accuracy_minus_confidence":
        metrics[
            "bias_accuracy_minus_confidence"
        ],

        "brier":
        metrics[
            "brier"
        ],

        "brier_skill":
        metrics[
            "brier_skill"
        ],

        "ece_fixed":
        metrics[
            "ece_fixed"
        ],

        "ece_adaptive":
        metrics[
            "ece_adaptive"
        ],

        "max_calibration_error":
        metrics[
            "max_calibration_error"
        ],
    }


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

    overall = calibration_metrics(
        frame
    )

    print("")
    print("=" * 100)
    print(
        "GLOBAL CONFIDENCE CALIBRATION"
    )
    print("=" * 100)

    print(
        f"accuracy="
        f"{overall['accuracy']:.4f}"
    )

    print(
        f"mean_confidence="
        f"{overall['mean_confidence']:.4f}"
    )

    print(
        f"bias_accuracy_minus_confidence="
        f"{overall['bias_accuracy_minus_confidence']:.4f}"
    )

    print(
        f"brier="
        f"{overall['brier']:.4f}"
    )

    print(
        f"baseline_brier="
        f"{overall['baseline_brier']:.4f}"
    )

    print(
        f"brier_skill="
        f"{overall['brier_skill']:.4f}"
    )

    print(
        f"ece_fixed="
        f"{overall['ece_fixed']:.4f}"
    )

    print(
        f"ece_adaptive="
        f"{overall['ece_adaptive']:.4f}"
    )

    print(
        f"max_calibration_error="
        f"{overall['max_calibration_error']:.4f}"
    )

    print("")
    print("=" * 100)
    print(
        "FIXED-WIDTH CALIBRATION BINS"
    )
    print("=" * 100)

    fixed_display = overall[
        "fixed_table"
    ].copy()

    numeric_columns = [
        "lower",
        "upper",
        "mean_confidence",
        "empirical_accuracy",
        "gap_accuracy_minus_confidence",
        "absolute_gap",
    ]

    fixed_display[
        numeric_columns
    ] = fixed_display[
        numeric_columns
    ].round(
        4
    )

    print(
        fixed_display.to_string(
            index=False
        )
    )

    print("")
    print("=" * 100)
    print(
        "ADAPTIVE CALIBRATION BINS"
    )
    print("=" * 100)

    adaptive_display = overall[
        "adaptive_table"
    ].copy()

    adaptive_display[
        numeric_columns
    ] = adaptive_display[
        numeric_columns
    ].round(
        4
    )

    print(
        adaptive_display.to_string(
            index=False
        )
    )

    rows = [
        summary_row(
            frame,
            "global",
            "all",
        )
    ]

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
            summary_row(
                part,
                "predicted_state",
                state,
            )
        )

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
            summary_row(
                part,
                "sport",
                sport,
            )
        )

    summary = pd.DataFrame(
        rows
    )

    print("")
    print("=" * 100)
    print(
        "CALIBRATION BY STATE AND SPORT"
    )
    print("=" * 100)

    print(
        summary.round(
            4
        ).to_string(
            index=False
        )
    )

    print("")
    print(
        "Running subject-level bootstrap..."
    )

    bootstrap = bootstrap_subjects(
        frame
    )

    print("")
    print("=" * 100)
    print(
        "SUBJECT-LEVEL BOOTSTRAP"
    )
    print("=" * 100)

    for column in [
        "brier",
        "brier_skill",
        "ece_fixed",
        "ece_adaptive",
        "bias",
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

    overall[
        "fixed_table"
    ].to_csv(
        OUTPUT_DIR
        / "confidence_calibration_bins_fixed.csv",
        index=False,
    )

    overall[
        "adaptive_table"
    ].to_csv(
        OUTPUT_DIR
        / "confidence_calibration_bins_adaptive.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "confidence_calibration_summary.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUTPUT_DIR
        / "confidence_calibration_bootstrap.csv",
        index=False,
    )

    print("")
    print(
        "saved="
        "results/baseline/"
        "confidence_calibration_bins_fixed.csv"
    )

    print(
        "saved="
        "results/baseline/"
        "confidence_calibration_bins_adaptive.csv"
    )

    print(
        "saved="
        "results/baseline/"
        "confidence_calibration_summary.csv"
    )

    print(
        "saved="
        "results/baseline/"
        "confidence_calibration_bootstrap.csv"
    )


if __name__ == "__main__":
    main()
