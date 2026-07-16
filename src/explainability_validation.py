import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILES = {
    "shap_anova": os.path.join(
        BASE_DIR,
        "outputs_explainability_kg",
        "shap_vs_anova_ranking.csv",
    ),
    "local_explanations": os.path.join(
        BASE_DIR,
        "outputs_explainability_kg",
        "local_explanations.csv",
    ),
    "separability": os.path.join(
        BASE_DIR,
        "outputs_separability",
        "separability_summary.csv",
    ),
    "model_comparison": os.path.join(
        BASE_DIR,
        "outputs_dl_multisport",
        "model_comparison_multisport.csv",
    ),
    "window_features": os.path.join(
        BASE_DIR,
        "outputs_dl_multisport",
        "window_features_multisport.csv",
    ),
}

FEATURE_TO_CONCEPT = {
    "mean_HR": "Heart rate",
    "std_HR": "Heart rate",
    "max_HR": "Heart rate",
    "mean_BR": "Breathing rate",
    "std_BR": "Breathing rate",
    "max_BR": "Breathing rate",
    "mean_RR": "RR interval",
    "std_RR": "RR interval",
    "min_RR": "RR interval",
}

CONCEPT_COLORS = {
    "Heart rate": "#E74C3C",
    "Breathing rate": "#3498DB",
    "RR interval": "#2ECC71",
}

STATE_ORDER = ["S0_Rest", "S1_Effort", "S2_Recovery"]
FLAG_ORDER = ["accept", "check", "review"]

FLAG_COLORS = {
    "accept": "#2ECC71",
    "check": "#F39C12",
    "review": "#E74C3C",
}


def check_input_files():
    missing = [
        path
        for path in INPUT_FILES.values()
        if not os.path.exists(path)
    ]

    if not missing:
        return

    print("Missing input files:")
    for path in missing:
        print(f"  {path}")

    raise FileNotFoundError(
        "Run the multisport, separability and explainability scripts first."
    )


def load_data():
    shap_anova = pd.read_csv(INPUT_FILES["shap_anova"])
    local_explanations = pd.read_csv(INPUT_FILES["local_explanations"])
    separability = pd.read_csv(INPUT_FILES["separability"])
    model_comparison = pd.read_csv(INPUT_FILES["model_comparison"])
    window_features = pd.read_csv(INPUT_FILES["window_features"])

    if "sport" not in separability.columns:
        first_column = separability.columns[0]
        separability = separability.rename(columns={first_column: "sport"})

    return {
        "shap_anova": shap_anova,
        "local_explanations": local_explanations,
        "separability": separability,
        "model_comparison": model_comparison,
        "window_features": window_features,
    }


def compute_shap_anova_metrics(shap_anova):
    required_columns = ["feature", "importance", "anova_f"]
    missing = [
        column
        for column in required_columns
        if column not in shap_anova.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in shap_vs_anova_ranking.csv: {missing}"
        )

    data = shap_anova.dropna(
        subset=["importance", "anova_f"]
    ).copy()

    if len(data) < 3:
        raise ValueError(
            "At least three valid features are required for correlation."
        )

    pearson_r, pearson_p = stats.pearsonr(
        data["anova_f"],
        data["importance"],
    )

    spearman_r, spearman_p = stats.spearmanr(
        data["anova_f"],
        data["importance"],
    )

    slope, intercept, regression_r, _, _ = stats.linregress(
        data["anova_f"],
        data["importance"],
    )

    metrics = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "r_squared": float(regression_r ** 2),
        "slope": float(slope),
        "intercept": float(intercept),
        "n_features": int(len(data)),
    }

    return data, metrics


def plot_shap_anova_correlation(shap_anova):
    data, metrics = compute_shap_anova_metrics(shap_anova)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for _, row in data.iterrows():
        feature = row["feature"]
        concept = FEATURE_TO_CONCEPT.get(feature, "Other")
        color = CONCEPT_COLORS.get(concept, "#7F8C8D")

        axes[0].scatter(
            row["anova_f"],
            row["importance"],
            s=100,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

        axes[0].annotate(
            feature,
            (row["anova_f"], row["importance"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    x_line = np.linspace(
        data["anova_f"].min(),
        data["anova_f"].max(),
        100,
    )

    y_line = (
        metrics["slope"] * x_line
        + metrics["intercept"]
    )

    axes[0].plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.3,
        color="gray",
    )

    axes[0].set_xlabel("Mean ANOVA F-score")
    axes[0].set_ylabel("Mean absolute SHAP importance")
    axes[0].set_title(
        "SHAP importance and statistical separability\n"
        f"Pearson r={metrics['pearson_r']:.3f}, "
        f"Spearman ρ={metrics['spearman_r']:.3f}"
    )
    axes[0].grid(alpha=0.3)

    handles = [
        mpatches.Patch(color=color, label=concept)
        for concept, color in CONCEPT_COLORS.items()
    ]
    axes[0].legend(handles=handles, fontsize=8)

    rank_data = data.copy()

    if "shap_rank" not in rank_data.columns:
        rank_data["shap_rank"] = (
            rank_data["importance"]
            .rank(ascending=False, method="first")
        )

    if "anova_rank" not in rank_data.columns:
        rank_data["anova_rank"] = (
            rank_data["anova_f"]
            .rank(ascending=False, method="first")
        )

    rank_data = rank_data.sort_values("shap_rank")
    positions = np.arange(len(rank_data))
    width = 0.36

    axes[1].bar(
        positions - width / 2,
        rank_data["shap_rank"],
        width,
        label="SHAP rank",
        alpha=0.85,
    )

    axes[1].bar(
        positions + width / 2,
        rank_data["anova_rank"],
        width,
        label="ANOVA rank",
        alpha=0.85,
    )

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        rank_data["feature"],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    axes[1].set_ylabel("Rank")
    axes[1].set_title("Feature ranking comparison")
    axes[1].invert_yaxis()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "1_shap_anova_correlation.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    return metrics


def plot_explanation_alignment(shap_anova):
    data = shap_anova.dropna(
        subset=["importance", "anova_f"]
    ).copy()

    data["concept"] = data["feature"].map(FEATURE_TO_CONCEPT)
    data["shap_normalized"] = (
        data["importance"]
        / data["importance"].max()
    )
    data["anova_normalized"] = (
        data["anova_f"]
        / data["anova_f"].max()
    )
    data["mean_alignment"] = (
        data["shap_normalized"]
        + data["anova_normalized"]
    ) / 2

    data = data.sort_values(
        "mean_alignment",
        ascending=False,
    )

    positions = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(
        positions + 0.18,
        data["shap_normalized"],
        height=0.32,
        label="SHAP importance",
        alpha=0.85,
    )

    ax.barh(
        positions - 0.18,
        data["anova_normalized"],
        height=0.32,
        label="ANOVA F-score",
        alpha=0.85,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(data["feature"])
    ax.set_xlim(0, 1.32)
    ax.set_xlabel("Normalized value")
    ax.set_title(
        "Alignment between model importance and statistical separability"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for position, (_, row) in zip(
        positions,
        data.iterrows(),
    ):
        concept = row["concept"]
        color = CONCEPT_COLORS.get(concept, "#7F8C8D")

        ax.text(
            1.04,
            position,
            concept,
            va="center",
            fontsize=8,
            color=color,
        )

    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "2_explanation_alignment.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    data.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_alignment.csv",
        ),
        index=False,
    )

    return data


def prepare_hitl_table(data, group_column):
    counts = (
        data
        .groupby([group_column, "hitl_flag"])
        .size()
        .unstack(fill_value=0)
    )

    for flag in FLAG_ORDER:
        if flag not in counts.columns:
            counts[flag] = 0

    counts = counts[FLAG_ORDER]

    totals = counts.sum(axis=1).replace(0, np.nan)
    percentages = counts.div(totals, axis=0)

    return counts, percentages


def draw_stacked_percentages(ax, percentages, title):
    bottom = np.zeros(len(percentages))

    for flag in FLAG_ORDER:
        values = percentages[flag].fillna(0).values

        ax.bar(
            range(len(percentages)),
            values,
            bottom=bottom,
            label=flag,
            color=FLAG_COLORS[flag],
            alpha=0.88,
        )

        for index, (value, base) in enumerate(
            zip(values, bottom)
        ):
            if value >= 0.06:
                text_color = (
                    "black"
                    if flag == "check"
                    else "white"
                )

                ax.text(
                    index,
                    base + value / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

        bottom += values

    ax.set_xticks(range(len(percentages)))
    ax.set_xticklabels(
        percentages.index,
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion of windows")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def plot_hitl_breakdown(local_explanations):
    required_columns = {
        "prediction",
        "sport",
        "hitl_flag",
        "kg_consistency",
    }

    missing = required_columns.difference(
        local_explanations.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns in local_explanations.csv: {sorted(missing)}"
        )

    data = local_explanations.copy()

    state_counts, state_percentages = prepare_hitl_table(
        data,
        "prediction",
    )

    state_order = [
        state
        for state in STATE_ORDER
        if state in state_percentages.index
    ]

    state_percentages = state_percentages.reindex(
        state_order
    )

    _, sport_percentages = prepare_hitl_table(
        data,
        "sport",
    )

    sport_percentages = sport_percentages.sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    draw_stacked_percentages(
        axes[0],
        state_percentages,
        "HITL decision by predicted state",
    )

    draw_stacked_percentages(
        axes[1],
        sport_percentages,
        "HITL decision by sport",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "3_hitl_breakdown.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    state_consistency = (
        data
        .groupby("prediction")["kg_consistency"]
        .agg(["mean", "std", "count"])
        .reindex(state_order)
    )

    sport_consistency = (
        data
        .groupby("sport")["kg_consistency"]
        .agg(["mean", "std", "count"])
        .sort_index()
    )

    state_consistency.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "kg_consistency_by_state.csv",
        )
    )

    sport_consistency.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "kg_consistency_by_sport.csv",
        )
    )

    return {
        "state_counts": state_counts,
        "state_consistency": state_consistency,
        "sport_consistency": sport_consistency,
    }


def count_dataset(window_features):
    n_windows = len(window_features)
    n_sports = window_features["sport"].nunique()
    n_subjects = window_features["subject"].nunique()
    n_states = window_features["true_state"].nunique()

    return {
        "n_windows": int(n_windows),
        "n_sports": int(n_sports),
        "n_subjects": int(n_subjects),
        "n_states": int(n_states),
    }


def select_model_scores(model_comparison):
    model_column = (
        "model"
        if "model" in model_comparison.columns
        else model_comparison.columns[0]
    )

    score_candidates = [
        "mean_loso_accuracy",
        "mean_accuracy",
        "accuracy",
    ]

    score_column = next(
        (
            column
            for column in score_candidates
            if column in model_comparison.columns
        ),
        None,
    )

    if score_column is None:
        raise ValueError(
            "No mean accuracy column found in model comparison file."
        )

    std_candidates = [
        "std_loso_accuracy",
        "std_accuracy",
        "accuracy_std",
    ]

    std_column = next(
        (
            column
            for column in std_candidates
            if column in model_comparison.columns
        ),
        None,
    )

    columns = [model_column, score_column]

    if std_column:
        columns.append(std_column)

    scores = model_comparison[columns].copy()

    scores = scores.rename(
        columns={
            model_column: "model",
            score_column: "mean_accuracy",
        }
    )

    if std_column:
        scores = scores.rename(
            columns={std_column: "std_accuracy"}
        )
    else:
        scores["std_accuracy"] = np.nan

    return scores.sort_values(
        "mean_accuracy",
        ascending=False,
    )


def plot_pipeline_summary(
    dataset_metrics,
    separability,
    model_scores,
    correlation_metrics,
    local_explanations,
):
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18, 6),
    )

    for ax in axes:
        ax.axis("off")

    axes[0].set_title("1. Dataset", fontweight="bold")
    axes[0].text(
        0.5,
        0.55,
        (
            f"{dataset_metrics['n_windows']} windows\n"
            f"{dataset_metrics['n_subjects']} groups\n"
            f"{dataset_metrics['n_sports']} sports\n"
            f"{dataset_metrics['n_states']} states"
        ),
        ha="center",
        va="center",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#ECF0F1",
            "edgecolor": "#7F8C8D",
        },
    )

    axes[1].set_title("2. Separability", fontweight="bold")

    valid_sep = separability.dropna(
        subset=["silhouette"]
    )

    separation_text = "\n".join(
        (
            f"{row['sport']}: "
            f"{row['silhouette']:+.3f}"
        )
        for _, row in valid_sep.iterrows()
    )

    axes[1].text(
        0.5,
        0.55,
        "Silhouette score\n\n" + separation_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#FEF9E7",
            "edgecolor": "#F39C12",
        },
    )

    axes[2].set_title("3. Prediction", fontweight="bold")

    model_text = "\n".join(
        (
            f"{row['model']}: "
            f"{row['mean_accuracy']:.3f}"
        )
        for _, row in model_scores.iterrows()
    )

    axes[2].text(
        0.5,
        0.55,
        "Mean LOSO accuracy\n\n" + model_text,
        ha="center",
        va="center",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#EBF5FB",
            "edgecolor": "#3498DB",
        },
    )

    axes[3].set_title("4. Explanation", fontweight="bold")

    axes[3].text(
        0.5,
        0.55,
        (
            "SHAP + ANOVA + KG\n\n"
            f"Pearson r = "
            f"{correlation_metrics['pearson_r']:.3f}\n"
            f"Spearman ρ = "
            f"{correlation_metrics['spearman_r']:.3f}\n"
            f"R² = "
            f"{correlation_metrics['r_squared']:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#F5EEF8",
            "edgecolor": "#8E44AD",
        },
    )

    axes[4].set_title("5. Human review", fontweight="bold")

    flag_percentages = (
        local_explanations["hitl_flag"]
        .value_counts(normalize=True)
        .reindex(FLAG_ORDER, fill_value=0)
    )

    hitl_text = "\n".join(
        (
            f"{flag}: {value:.1%}"
        )
        for flag, value in flag_percentages.items()
    )

    mean_consistency = (
        local_explanations["kg_consistency"].mean()
    )

    axes[4].text(
        0.5,
        0.55,
        (
            hitl_text
            + f"\n\nMean KG consistency: "
            f"{mean_consistency:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#E8F8F5",
            "edgecolor": "#27AE60",
        },
    )

    for index in range(4):
        axes[index].annotate(
            "",
            xy=(1.05, 0.55),
            xytext=(0.95, 0.55),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "color": "#7F8C8D",
                "linewidth": 1.5,
            },
        )

    fig.suptitle(
        "Validation workflow for the explainable multi-sport pipeline",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "4_pipeline_summary.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def save_validation_summary(
    correlation_metrics,
    separability,
    model_scores,
    local_explanations,
):
    rows = [
        {
            "section": "SHAP_ANOVA",
            "metric": "pearson_r",
            "value": correlation_metrics["pearson_r"],
        },
        {
            "section": "SHAP_ANOVA",
            "metric": "pearson_p",
            "value": correlation_metrics["pearson_p"],
        },
        {
            "section": "SHAP_ANOVA",
            "metric": "spearman_r",
            "value": correlation_metrics["spearman_r"],
        },
        {
            "section": "SHAP_ANOVA",
            "metric": "spearman_p",
            "value": correlation_metrics["spearman_p"],
        },
        {
            "section": "SHAP_ANOVA",
            "metric": "r_squared",
            "value": correlation_metrics["r_squared"],
        },
        {
            "section": "KG",
            "metric": "mean_consistency",
            "value": local_explanations[
                "kg_consistency"
            ].mean(),
        },
    ]

    for _, row in separability.iterrows():
        sport = row.get("sport")

        if pd.notna(row.get("silhouette")):
            rows.append({
                "section": "Separability",
                "metric": f"silhouette_{sport}",
                "value": row["silhouette"],
            })

        if pd.notna(row.get("fisher_mean_HR")):
            rows.append({
                "section": "Separability",
                "metric": f"fisher_mean_HR_{sport}",
                "value": row["fisher_mean_HR"],
            })

    for _, row in model_scores.iterrows():
        rows.append({
            "section": "Prediction",
            "metric": f"mean_accuracy_{row['model']}",
            "value": row["mean_accuracy"],
        })

    flag_percentages = (
        local_explanations["hitl_flag"]
        .value_counts(normalize=True)
        .reindex(FLAG_ORDER, fill_value=0)
    )

    for flag, value in flag_percentages.items():
        rows.append({
            "section": "HITL",
            "metric": f"{flag}_proportion",
            "value": value,
        })

    summary = pd.DataFrame(rows)
    summary["value"] = summary["value"].astype(float).round(4)

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_summary.csv",
        ),
        index=False,
    )

    return summary


def main():
    print("=" * 68)
    print("XAI validation")
    print("No model is trained in this script.")
    print("=" * 68)

    check_input_files()
    data = load_data()

    shap_anova = data["shap_anova"]
    local_explanations = data["local_explanations"]
    separability = data["separability"]
    model_comparison = data["model_comparison"]
    window_features = data["window_features"]

    print("")
    print("Loaded files")
    print(f"  SHAP/ANOVA features: {len(shap_anova)}")
    print(f"  Local explanations:  {len(local_explanations)}")
    print(f"  Sports:              {len(separability)}")
    print(f"  Model rows:          {len(model_comparison)}")

    print("")
    print("[1] SHAP and ANOVA comparison")
    correlation_metrics = plot_shap_anova_correlation(
        shap_anova
    )

    print(
        f"  Pearson r = "
        f"{correlation_metrics['pearson_r']:.3f} "
        f"(p={correlation_metrics['pearson_p']:.4f})"
    )
    print(
        f"  Spearman rho = "
        f"{correlation_metrics['spearman_r']:.3f} "
        f"(p={correlation_metrics['spearman_p']:.4f})"
    )
    print(
        f"  R squared = "
        f"{correlation_metrics['r_squared']:.3f}"
    )

    print("")
    print("[2] Feature alignment")
    alignment = plot_explanation_alignment(shap_anova)
    print(
        alignment[
            [
                "feature",
                "shap_normalized",
                "anova_normalized",
                "concept",
            ]
        ]
        .head(6)
        .round(3)
        .to_string(index=False)
    )

    print("")
    print("[3] HITL and KG consistency")
    hitl_results = plot_hitl_breakdown(
        local_explanations
    )

    print("")
    print("Mean KG consistency by state")
    print(
        hitl_results["state_consistency"]
        .round(3)
        .to_string()
    )

    dataset_metrics = count_dataset(window_features)
    model_scores = select_model_scores(model_comparison)

    print("")
    print("[4] Pipeline summary")
    plot_pipeline_summary(
        dataset_metrics,
        separability,
        model_scores,
        correlation_metrics,
        local_explanations,
    )

    print("")
    print("[5] Validation summary")
    summary = save_validation_summary(
        correlation_metrics,
        separability,
        model_scores,
        local_explanations,
    )
    print(summary.to_string(index=False))

    print("")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("  1_shap_anova_correlation.png")
    print("  2_explanation_alignment.png")
    print("  3_hitl_breakdown.png")
    print("  4_pipeline_summary.png")
    print("  feature_alignment.csv")
    print("  kg_consistency_by_state.csv")
    print("  kg_consistency_by_sport.csv")
    print("  validation_summary.csv")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()