import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs_dl_multisport",
    "window_features_multisport.csv",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_separability")
os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURE_NAMES = [
    "mean_HR", "std_HR", "max_HR",
    "mean_BR", "std_BR", "max_BR",
    "mean_RR", "std_RR", "min_RR",
]

STATE_ORDER = ["S0_Rest", "S1_Effort", "S2_Recovery"]
SPORTS = ["RUN", "CYC", "TRE", "SKA"]

STATE_COLORS = {
    "S0_Rest": "#3498DB",
    "S1_Effort": "#E74C3C",
    "S2_Recovery": "#2ECC71",
}


def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            "window_features_multisport.csv not found. "
            "Run deep_learning_multisport.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded windows: {len(df)}")
    print(f"Sports: {sorted(df['sport'].unique())}")
    print(f"States: {sorted(df['true_state'].unique())}")

    return df


def compute_anova_scores(df):
    rows = []

    for sport in SPORTS:
        sport_df = df[df["sport"] == sport]
        row = {"sport": sport}

        for feature in FEATURE_NAMES:
            groups = []

            for state in STATE_ORDER:
                values = sport_df[sport_df["true_state"] == state][feature].dropna().values

                if len(values) > 1:
                    groups.append(values)

            if len(groups) >= 2:
                f_score, p_value = stats.f_oneway(*groups)
                row[feature] = float(f_score)
            else:
                row[feature] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("sport")


def plot_anova_heatmap(anova_df):
    matrix = anova_df[FEATURE_NAMES].values.astype(float)
    vmax = np.nanpercentile(matrix, 90)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    image = ax.imshow(
        matrix,
        cmap="YlOrRd",
        aspect="auto",
        vmin=0,
        vmax=vmax,
    )

    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(len(anova_df.index)))
    ax.set_yticklabels(anova_df.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]

            if np.isnan(value):
                text = "N/A"
                color = "gray"
            else:
                text = f"{value:.0f}"
                color = "white" if value > vmax * 0.6 else "black"

            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(image, ax=ax, label="ANOVA F-score")
    ax.set_title("Feature separability by sport")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "1_anova_fscores.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_hr_distributions(df):
    fig, axes = plt.subplots(1, len(SPORTS), figsize=(14, 5), sharey=False)

    for ax, sport in zip(axes, SPORTS):
        sport_df = df[df["sport"] == sport]

        data = []
        labels = []
        colors = []

        for state in STATE_ORDER:
            values = sport_df[sport_df["true_state"] == state]["mean_HR"].dropna().values

            if len(values) == 0:
                continue

            data.append(values)
            labels.append(state.replace("S0_", "").replace("S1_", "").replace("S2_", ""))
            colors.append(STATE_COLORS[state])

        if data:
            box = ax.boxplot(
                data,
                patch_artist=True,
                showfliers=False,
                widths=0.55,
            )

            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            for median in box["medians"]:
                median.set_color("white")
                median.set_linewidth(2)

        ax.set_title(sport)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("mean HR (bpm)" if sport == SPORTS[0] else "")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "2_hr_distributions.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def compute_silhouette_scores(df):
    rows = []

    for sport in SPORTS:
        sport_df = df[df["sport"] == sport].dropna(subset=FEATURE_NAMES)

        if sport_df["true_state"].nunique() < 2 or len(sport_df) < 10:
            rows.append({
                "sport": sport,
                "silhouette": np.nan,
                "n_windows": len(sport_df),
            })
            continue

        x = sport_df[FEATURE_NAMES].values
        x = StandardScaler().fit_transform(x)

        y = LabelEncoder().fit_transform(sport_df["true_state"])

        score = silhouette_score(
            x,
            y,
            sample_size=min(5000, len(sport_df)),
            random_state=42,
        )

        rows.append({
            "sport": sport,
            "silhouette": float(score),
            "n_windows": len(sport_df),
        })

    return pd.DataFrame(rows)


def plot_silhouette_scores(silhouette_df):
    valid = silhouette_df.dropna(subset=["silhouette"])

    colors = []
    for value in valid["silhouette"]:
        if value < 0:
            colors.append("#E74C3C")
        elif value < 0.25:
            colors.append("#F39C12")
        elif value < 0.45:
            colors.append("#3498DB")
        else:
            colors.append("#2ECC71")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        valid["sport"],
        valid["silhouette"],
        color=colors,
        edgecolor="white",
        width=0.55,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.25, color="gray", linestyle=":", linewidth=1)

    for bar, value in zip(bars, valid["silhouette"]):
        y_pos = value + 0.02 if value >= 0 else value - 0.04
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{value:+.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(-0.2, 0.65)
    ax.set_ylabel("Silhouette score")
    ax.set_title("State separation by sport")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "3_silhouette_scores.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def fisher_score(values_by_state):
    if len(values_by_state) < 2:
        return np.nan

    all_values = np.concatenate(values_by_state)
    global_mean = all_values.mean()

    between = np.mean([(values.mean() - global_mean) ** 2 for values in values_by_state])
    within = np.mean([values.std() ** 2 for values in values_by_state])

    return float(between / (within + 1e-9))


def compute_fisher_hr(df):
    rows = []

    for sport in SPORTS:
        sport_df = df[df["sport"] == sport]
        values_by_state = []

        for state in STATE_ORDER:
            values = sport_df[sport_df["true_state"] == state]["mean_HR"].dropna().values

            if len(values) > 1:
                values_by_state.append(values)

        rows.append({
            "sport": sport,
            "fisher_mean_HR": fisher_score(values_by_state),
        })

    return pd.DataFrame(rows)


def plot_fisher_hr(df, fisher_df):
    fig, axes = plt.subplots(1, len(SPORTS), figsize=(14, 4.5), sharey=False)

    for ax, sport in zip(axes, SPORTS):
        sport_df = df[df["sport"] == sport]
        fisher_value = fisher_df[fisher_df["sport"] == sport]["fisher_mean_HR"].iloc[0]

        labels = []
        means = []
        stds = []
        colors = []

        for state in STATE_ORDER:
            values = sport_df[sport_df["true_state"] == state]["mean_HR"].dropna().values

            if len(values) == 0:
                continue

            labels.append(state.replace("S0_", "").replace("S1_", "").replace("S2_", ""))
            means.append(values.mean())
            stds.append(values.std())
            colors.append(STATE_COLORS[state])

        x = np.arange(len(labels))

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.8,
            width=0.6,
        )

        for i, mean_value in enumerate(means):
            ax.text(i, mean_value + stds[i] + 2, f"{mean_value:.0f}", ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"{sport}\nFisher HR = {fisher_value:.2f}")
        ax.set_ylabel("mean HR (bpm)" if sport == SPORTS[0] else "")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "4_fisher_hr.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def feature_ranking_per_sport(df):
    rows = []

    for sport in SPORTS:
        sport_df = df[df["sport"] == sport]

        for feature in FEATURE_NAMES:
            groups = []

            for state in STATE_ORDER:
                values = sport_df[sport_df["true_state"] == state][feature].dropna().values

                if len(values) > 1:
                    groups.append(values)

            if len(groups) >= 2:
                f_score, p_value = stats.f_oneway(*groups)
                discriminant = "high" if f_score > 100 else "medium" if f_score > 10 else "low"
            else:
                f_score = np.nan
                p_value = np.nan
                discriminant = "n/a"

            rows.append({
                "sport": sport,
                "feature": feature,
                "anova_f": float(f_score) if not np.isnan(f_score) else np.nan,
                "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                "discriminant": discriminant,
            })

    ranking_df = pd.DataFrame(rows)

    ranking_df["rank"] = (
        ranking_df
        .groupby("sport")["anova_f"]
        .rank(ascending=False, method="first")
    )

    ranking_df = ranking_df.sort_values(["sport", "rank"])
    ranking_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_ranking_per_sport.csv"),
        index=False,
    )

    return ranking_df


def plot_feature_ranking_heatmap(ranking_df):
    pivot = ranking_df.pivot(
        index="feature",
        columns="sport",
        values="anova_f",
    )

    available_sports = [sport for sport in SPORTS if sport in pivot.columns]
    pivot = pivot[available_sports]

    if "RUN" in pivot.columns:
        pivot = pivot.sort_values("RUN", ascending=False)

    values = pivot.values.astype(float)
    log_values = np.log1p(values)

    fig, ax = plt.subplots(figsize=(9, 6))

    image = ax.imshow(log_values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    max_log = np.nanmax(log_values)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]

            if np.isnan(value):
                text = "—"
                color = "gray"
            else:
                text = f"{value:.0f}"
                color = "white" if log_values[i, j] > max_log * 0.6 else "black"

            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(image, ax=ax, label="log(1 + F-score)")
    ax.set_title("Feature ranking by sport")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "5_feature_ranking_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_top_features(df, ranking_df):
    sports_with_features = []

    for sport in SPORTS:
        sub_rank = ranking_df[
            (ranking_df["sport"] == sport)
            & ranking_df["anova_f"].notna()
        ]

        if len(sub_rank) > 0:
            sports_with_features.append(sport)

    fig, axes = plt.subplots(
        len(sports_with_features),
        3,
        figsize=(14, 4 * len(sports_with_features)),
    )

    if len(sports_with_features) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, sport in enumerate(sports_with_features):
        sport_df = df[df["sport"] == sport]

        top_features = (
            ranking_df[
                (ranking_df["sport"] == sport)
                & ranking_df["anova_f"].notna()
            ]
            .sort_values("anova_f", ascending=False)
            .head(3)
        )

        for col_idx, (_, row) in enumerate(top_features.iterrows()):
            ax = axes[row_idx, col_idx]
            feature = row["feature"]

            for state in STATE_ORDER:
                values = sport_df[sport_df["true_state"] == state][feature].dropna().values

                if len(values) > 0:
                    ax.hist(
                        values,
                        bins=25,
                        alpha=0.55,
                        color=STATE_COLORS[state],
                        label=state,
                        density=True,
                    )

            ax.set_title(f"{sport} - {feature}\nF={row['anova_f']:.1f}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Density" if col_idx == 0 else "")
            ax.grid(alpha=0.3)

            if row_idx == 0 and col_idx == 2:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "6_top3_features_distributions.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def save_summary(anova_df, silhouette_df, fisher_df):
    anova_mean = anova_df.mean(axis=1).rename("mean_anova_fscore")

    summary = (
        silhouette_df.set_index("sport")
        .join(fisher_df.set_index("sport"))
        .join(anova_mean)
    )

    summary.to_csv(os.path.join(OUTPUT_DIR, "separability_summary.csv"))

    return summary


def print_hr_stats(df):
    print("")
    print("Mean HR by sport and state")

    for sport in SPORTS:
        sport_df = df[df["sport"] == sport]

        if len(sport_df) == 0:
            continue

        print(f"\n{sport}")

        for state in STATE_ORDER:
            values = sport_df[sport_df["true_state"] == state]["mean_HR"].dropna().values

            if len(values) == 0:
                continue

            print(
                f"  {state:<15} "
                f"mean={values.mean():6.1f} "
                f"std={values.std():5.1f} "
                f"n={len(values)}"
            )


def print_feature_ranking(ranking_df):
    print("")
    print("Top features by sport")

    for sport in SPORTS:
        sub_rank = ranking_df[
            (ranking_df["sport"] == sport)
            & ranking_df["anova_f"].notna()
        ]

        if len(sub_rank) == 0:
            print(f"  {sport}: no ranking available")
            continue

        top = sub_rank.sort_values("anova_f", ascending=False).head(3)
        features = ", ".join(top["feature"].tolist())
        max_f = top.iloc[0]["anova_f"]
        min_f = sub_rank["anova_f"].min()

        print(f"  {sport}: {features}  (max F={max_f:.1f}, min F={min_f:.1f})")


def main():
    df = load_data()

    print_hr_stats(df)

    anova_df = compute_anova_scores(df)
    silhouette_df = compute_silhouette_scores(df)
    fisher_df = compute_fisher_hr(df)
    ranking_df = feature_ranking_per_sport(df)

    print("")
    print("ANOVA F-scores")
    print(anova_df.round(2).to_string())

    print("")
    print("Silhouette scores")
    print(silhouette_df.round(3).to_string(index=False))

    print("")
    print("Fisher score on mean_HR")
    print(fisher_df.round(3).to_string(index=False))

    print_feature_ranking(ranking_df)

    plot_anova_heatmap(anova_df)
    plot_hr_distributions(df)
    plot_silhouette_scores(silhouette_df)
    plot_fisher_hr(df, fisher_df)
    plot_feature_ranking_heatmap(ranking_df)
    plot_top_features(df, ranking_df)

    summary = save_summary(anova_df, silhouette_df, fisher_df)

    print("")
    print("Summary")
    print(summary.round(3).to_string())

    print("")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("  1_anova_fscores.png")
    print("  2_hr_distributions.png")
    print("  3_silhouette_scores.png")
    print("  4_fisher_hr.png")
    print("  5_feature_ranking_heatmap.png")
    print("  6_top3_features_distributions.png")
    print("  separability_summary.csv")
    print("  feature_ranking_per_sport.csv")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()