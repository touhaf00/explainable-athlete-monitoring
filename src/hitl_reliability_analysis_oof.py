from __future__ import annotations

import os
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs_explainability_kg",
    "local_explanations_oof.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs_hitl_reliability",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

HITL_ORDER = ["accept", "check", "review"]


def load_explanations() -> pd.DataFrame:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            "local_explanations.csv was not found. "
            "Run explainability_kg.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "sport",
        "subject",
        "true_state",
        "prediction",
        "confidence",
        "kg_consistency",
        "hitl_flag",
    }

    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            "Missing columns in local_explanations.csv: "
            + ", ".join(missing_columns)
        )

    df = df.dropna(
        subset=[
            "true_state",
            "prediction",
            "confidence",
            "kg_consistency",
            "hitl_flag",
        ]
    ).copy()

    df["sport"] = df["sport"].astype(str)
    df["subject"] = df["subject"].astype(str)
    df["true_state"] = df["true_state"].astype(str)
    df["prediction"] = df["prediction"].astype(str)
    df["hitl_flag"] = df["hitl_flag"].astype(str)

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce",
    )

    df["kg_consistency"] = pd.to_numeric(
        df["kg_consistency"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "confidence",
            "kg_consistency",
        ]
    ).copy()

    df["confidence"] = df["confidence"].clip(0.0, 1.0)
    df["kg_consistency"] = df["kg_consistency"].clip(0.0, 1.0)

    df["correct"] = (
        df["true_state"] == df["prediction"]
    ).astype(int)

    df["reliability_score"] = np.sqrt(
        df["confidence"] * df["kg_consistency"]
    )

    if df.empty:
        raise ValueError(
            "No valid rows remain after cleaning local_explanations.csv."
        )

    return df.reset_index(drop=True)


def wilson_interval(
    correct: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan

    proportion = correct / total
    denominator = 1 + (z ** 2 / total)

    center = (
        proportion
        + (z ** 2 / (2 * total))
    ) / denominator

    margin = (
        z
        * sqrt(
            (
                proportion * (1 - proportion) / total
            )
            + (
                z ** 2 / (4 * total ** 2)
            )
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_group(
    group: pd.DataFrame,
) -> dict[str, float | int]:
    total = len(group)
    correct = int(group["correct"].sum())
    accuracy = correct / total if total else np.nan
    lower, upper = wilson_interval(correct, total)

    return {
        "n_windows": total,
        "n_correct": correct,
        "accuracy": accuracy,
        "risk": 1.0 - accuracy if total else np.nan,
        "accuracy_ci_low": lower,
        "accuracy_ci_high": upper,
        "mean_confidence": float(group["confidence"].mean()),
        "mean_consistency": float(group["kg_consistency"].mean()),
        "mean_reliability": float(group["reliability_score"].mean()),
    }


def build_hitl_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for flag in HITL_ORDER:
        group = df[df["hitl_flag"] == flag]

        if group.empty:
            continue

        rows.append({
            "hitl_flag": flag,
            **summarize_group(group),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["coverage"] = result["n_windows"] / len(df)

    return result


def build_grouped_summary(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows = []

    for group_value in sorted(df[group_column].unique()):
        group_df = df[df[group_column] == group_value]

        for flag in HITL_ORDER:
            flag_df = group_df[group_df["hitl_flag"] == flag]

            if flag_df.empty:
                continue

            row = {
                group_column: group_value,
                "hitl_flag": flag,
                **summarize_group(flag_df),
            }

            row["coverage_within_group"] = len(flag_df) / len(group_df)
            rows.append(row)

    return pd.DataFrame(rows)


def build_risk_coverage_curve(
    df: pd.DataFrame,
) -> pd.DataFrame:
    ranked = df.sort_values(
        "reliability_score",
        ascending=False,
    ).reset_index(drop=True)

    rows = []

    for coverage in np.linspace(0.05, 1.0, 20):
        n_selected = max(
            1,
            int(round(coverage * len(ranked))),
        )

        selected = ranked.iloc[:n_selected]
        correct = int(selected["correct"].sum())
        accuracy = correct / n_selected
        lower, upper = wilson_interval(correct, n_selected)

        rows.append({
            "coverage": n_selected / len(ranked),
            "n_selected": n_selected,
            "accuracy": accuracy,
            "risk": 1.0 - accuracy,
            "accuracy_ci_low": lower,
            "accuracy_ci_high": upper,
            "minimum_reliability": float(
                selected["reliability_score"].min()
            ),
            "mean_confidence": float(
                selected["confidence"].mean()
            ),
            "mean_consistency": float(
                selected["kg_consistency"].mean()
            ),
        })

    return pd.DataFrame(rows)


def build_threshold_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for threshold in np.linspace(0.20, 0.90, 15):
        selected = df[
            df["reliability_score"] >= threshold
        ]

        if selected.empty:
            continue

        correct = int(selected["correct"].sum())
        total = len(selected)
        accuracy = correct / total
        lower, upper = wilson_interval(correct, total)

        rows.append({
            "threshold": threshold,
            "coverage": total / len(df),
            "n_selected": total,
            "accuracy": accuracy,
            "risk": 1.0 - accuracy,
            "accuracy_ci_low": lower,
            "accuracy_ci_high": upper,
        })

    return pd.DataFrame(rows)


def plot_accuracy_by_hitl(
    summary: pd.DataFrame,
) -> None:
    if summary.empty:
        return

    summary = (
        summary
        .set_index("hitl_flag")
        .reindex(HITL_ORDER)
        .dropna(how="all")
        .reset_index()
    )

    values = summary["accuracy"].to_numpy()
    lower_errors = values - summary["accuracy_ci_low"].to_numpy()
    upper_errors = summary["accuracy_ci_high"].to_numpy() - values

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        summary["hitl_flag"],
        values,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=5,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("HITL decision")
    ax.set_title("Prediction accuracy by HITL decision")

    for bar, accuracy, count in zip(
        bars,
        summary["accuracy"],
        summary["n_windows"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.02, accuracy + 0.04),
            f"{accuracy:.1%}\n(n={int(count)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "1_accuracy_by_hitl.png",
        ),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_accuracy_by_sport(
    sport_summary: pd.DataFrame,
) -> None:
    if sport_summary.empty:
        return

    pivot = sport_summary.pivot(
        index="sport",
        columns="hitl_flag",
        values="accuracy",
    ).reindex(columns=HITL_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))

    pivot.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Sport")
    ax.set_title("HITL reliability by sport")
    ax.legend(title="HITL decision")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "2_accuracy_by_sport_and_hitl.png",
        ),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_risk_coverage(
    curve: pd.DataFrame,
) -> None:
    if curve.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        curve["coverage"],
        curve["risk"],
        marker="o",
        linewidth=1.8,
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(
        0,
        min(
            1.0,
            max(0.05, curve["risk"].max() + 0.05),
        ),
    )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk among retained predictions")
    ax.set_title("Risk-coverage curve")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "3_risk_coverage_curve.png",
        ),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_coverage_accuracy(
    curve: pd.DataFrame,
) -> None:
    if curve.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        curve["coverage"],
        curve["accuracy"],
        marker="o",
        linewidth=1.8,
    )

    ax.fill_between(
        curve["coverage"],
        curve["accuracy_ci_low"],
        curve["accuracy_ci_high"],
        alpha=0.2,
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy among retained predictions")
    ax.set_title("Coverage-accuracy trade-off")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "4_coverage_accuracy_tradeoff.png",
        ),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_main_results(
    hitl_summary: pd.DataFrame,
    sport_summary: pd.DataFrame,
    risk_curve: pd.DataFrame,
) -> None:
    print("HITL reliability analysis")
    print(f"Windows: {int(hitl_summary['n_windows'].sum())}")
    print("")

    print("Accuracy by HITL decision")

    print(
        hitl_summary[
            [
                "hitl_flag",
                "n_windows",
                "coverage",
                "accuracy",
                "risk",
                "mean_confidence",
                "mean_consistency",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("")
    print("Accuracy by sport and HITL decision")

    print(
        sport_summary[
            [
                "sport",
                "hitl_flag",
                "n_windows",
                "coverage_within_group",
                "accuracy",
                "mean_confidence",
                "mean_consistency",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("")
    print("Selected risk-coverage points")

    selected_points = risk_curve.iloc[
        [0, 4, 9, 14, 19]
    ][
        [
            "coverage",
            "n_selected",
            "accuracy",
            "risk",
            "minimum_reliability",
        ]
    ]

    print(
        selected_points
        .round(4)
        .to_string(index=False)
    )


def main() -> None:
    df = load_explanations()

    hitl_summary = build_hitl_summary(df)
    sport_summary = build_grouped_summary(
        df,
        "sport",
    )
    state_summary = build_grouped_summary(
        df,
        "prediction",
    )
    risk_curve = build_risk_coverage_curve(df)
    threshold_summary = build_threshold_summary(df)

    hitl_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "hitl_reliability_summary.csv",
        ),
        index=False,
    )

    sport_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "hitl_reliability_by_sport.csv",
        ),
        index=False,
    )

    state_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "hitl_reliability_by_state.csv",
        ),
        index=False,
    )

    risk_curve.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "risk_coverage_curve.csv",
        ),
        index=False,
    )

    threshold_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "reliability_threshold_summary.csv",
        ),
        index=False,
    )

    plot_accuracy_by_hitl(hitl_summary)
    plot_accuracy_by_sport(sport_summary)
    plot_risk_coverage(risk_curve)
    plot_coverage_accuracy(risk_curve)

    print_main_results(
        hitl_summary=hitl_summary,
        sport_summary=sport_summary,
        risk_curve=risk_curve,
    )

    print("")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("  1_accuracy_by_hitl.png")
    print("  2_accuracy_by_sport_and_hitl.png")
    print("  3_risk_coverage_curve.png")
    print("  4_coverage_accuracy_tradeoff.png")
    print("  hitl_reliability_summary.csv")
    print("  hitl_reliability_by_sport.csv")
    print("  hitl_reliability_by_state.csv")
    print("  risk_coverage_curve.csv")
    print("  reliability_threshold_summary.csv")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
