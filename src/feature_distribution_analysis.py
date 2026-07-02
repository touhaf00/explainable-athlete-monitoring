import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs_dl_multisport",
    "window_features_multisport.csv",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_distribution_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "mean_HR", "std_HR", "max_HR",
    "mean_BR", "std_BR", "max_BR",
    "mean_RR", "std_RR", "min_RR",
]

STATE_ORDER = ["S0_Rest", "S1_Effort", "S2_Recovery"]

STATE_COLORS = {
    "S0_Rest": "#3498DB",
    "S1_Effort": "#E74C3C",
    "S2_Recovery": "#2ECC71",
}

SPORT_COLORS = {
    "RUN": "#3498DB",
    "CYC": "#2ECC71",
    "TRE": "#E74C3C",
    "SKA": "#9B59B6",
}


def save_distribution_plot(df):
    features = ["mean_HR", "mean_BR", "mean_RR"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, feature in zip(axes, features):
        values = []
        labels = []
        colors = []
        positions = []
        position = 0

        for sport in sorted(df["sport"].unique()):
            for state in STATE_ORDER:
                subset = df[(df["sport"] == sport) & (df["true_state"] == state)]

                if len(subset) == 0:
                    continue

                values.append(subset[feature].values)
                labels.append(f"{sport}\n{state.replace('_', ' ')}")
                colors.append(STATE_COLORS[state])
                positions.append(position)
                position += 1

            position += 0.5

        boxplot = ax.boxplot(
            values,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            showfliers=False,
        )

        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "1_distributions_by_sport_state.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def print_basic_statistics(df):
    print("")
    print("Mean HR by sport and state")

    for sport in sorted(df["sport"].unique()):
        for state in STATE_ORDER:
            subset = df[(df["sport"] == sport) & (df["true_state"] == state)]

            if len(subset) == 0:
                continue

            mean_hr = subset["mean_HR"].mean()
            std_hr = subset["mean_HR"].std()

            print(f"  {sport:<5} {state:<15}: {mean_hr:6.1f} +/- {std_hr:5.1f}  n={len(subset)}")


def save_pca_plot(df):
    x = df[FEATURE_NAMES].values
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)

    df = df.copy()
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]

    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for state in STATE_ORDER:
        subset = df[df["true_state"] == state]
        axes[0].scatter(
            subset["pca_x"],
            subset["pca_y"],
            s=6,
            alpha=0.4,
            color=STATE_COLORS[state],
            label=state,
        )

    axes[0].set_xlabel(f"PC1 ({explained[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({explained[1]:.1%})")
    axes[0].set_title("PCA by state")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for sport in sorted(df["sport"].unique()):
        subset = df[df["sport"] == sport]
        axes[1].scatter(
            subset["pca_x"],
            subset["pca_y"],
            s=6,
            alpha=0.4,
            color=SPORT_COLORS.get(sport, "#777777"),
            label=sport,
        )

    axes[1].set_xlabel(f"PC1 ({explained[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({explained[1]:.1%})")
    axes[1].set_title("PCA by sport")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "2_pca_projection.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    return df, explained


def save_umap_plot(df):
    if not UMAP_AVAILABLE:
        print("")
        print("UMAP not available. Install it with: pip3 install umap-learn")
        return

    x = df[FEATURE_NAMES].values
    x_scaled = StandardScaler().fit_transform(x)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )

    coords = reducer.fit_transform(x_scaled)

    df = df.copy()
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for state in STATE_ORDER:
        subset = df[df["true_state"] == state]
        axes[0].scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=6,
            alpha=0.4,
            color=STATE_COLORS[state],
            label=state,
        )

    axes[0].set_title("UMAP by state")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for sport in sorted(df["sport"].unique()):
        subset = df[df["sport"] == sport]
        axes[1].scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=6,
            alpha=0.4,
            color=SPORT_COLORS.get(sport, "#777777"),
            label=sport,
        )

    axes[1].set_title("UMAP by sport")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "3_umap_projection.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def print_silhouette_scores(df):
    print("")
    print("State separation in PCA space")

    for sport in sorted(df["sport"].unique()):
        subset = df[df["sport"] == sport]

        if subset["true_state"].nunique() < 2:
            continue

        try:
            score = silhouette_score(
                subset[["pca_x", "pca_y"]],
                subset["true_state"],
            )
            print(f"  {sport:<5}: silhouette={score:+.3f}")
        except Exception:
            continue


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            "window_features_multisport.csv was not found. "
            "Run deep_learning_multisport.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df)} windows")
    print(f"Sports: {sorted(df['sport'].unique())}")

    save_distribution_plot(df)
    print_basic_statistics(df)

    df_pca, explained = save_pca_plot(df)

    print("")
    print(f"PCA variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

    save_umap_plot(df)
    print_silhouette_scores(df_pca)

    print("")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("  1_distributions_by_sport_state.png")
    print("  2_pca_projection.png")
    print("  3_umap_projection.png if UMAP is installed")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()