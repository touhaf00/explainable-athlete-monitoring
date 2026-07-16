from __future__ import annotations

import os
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError as error:
    raise ImportError(
        "The SHAP package is required for this script. "
        "Install it with: pip3 install shap"
    ) from error


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs_dl_multisport",
    "window_features_multisport.csv",
)

RANKING_FILE = os.path.join(
    BASE_DIR,
    "outputs_separability",
    "feature_ranking_per_sport.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs_explainability_kg",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURES = [
    "mean_HR",
    "std_HR",
    "max_HR",
    "mean_BR",
    "std_BR",
    "max_BR",
    "mean_RR",
    "std_RR",
    "min_RR",
]

STATES = [
    "S0_Rest",
    "S1_Effort",
    "S2_Recovery",
]

STATE_TO_ID = {
    state: index
    for index, state in enumerate(STATES)
}

ID_TO_STATE = {
    index: state
    for state, index in STATE_TO_ID.items()
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


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            "window_features_multisport.csv was not found. "
            "Run deep_learning_multisport.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "window_id",
        "sport",
        "subject",
        "true_state",
        *FEATURES,
    }

    missing_columns = sorted(
        required_columns.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in window_features_multisport.csv: "
            + ", ".join(missing_columns)
        )

    df = df[df["true_state"].isin(STATES)].copy()
    df = df.dropna(subset=FEATURES)
    df["label"] = df["true_state"].map(STATE_TO_ID)

    if df.empty:
        raise ValueError(
            "No valid windows remain after filtering."
        )

    df["window_id"] = df["window_id"].astype(int)
    df["sport"] = df["sport"].astype(str)
    df["subject"] = df["subject"].astype(str)
    df["true_state"] = df["true_state"].astype(str)
    df["label"] = df["label"].astype(int)

    print(f"Loaded windows: {len(df)}")
    print(f"Sports: {sorted(df['sport'].unique())}")
    print(f"States: {sorted(df['true_state'].unique())}")

    return df.reset_index(drop=True)


def make_model(
    n_estimators: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def train_random_forest(
    df: pd.DataFrame,
) -> tuple[
    RandomForestClassifier,
    StandardScaler,
    np.ndarray,
    np.ndarray,
]:
    x = df[FEATURES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["subject"].to_numpy(dtype=str)

    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        raise ValueError(
            "At least two groups are required for grouped validation."
        )

    n_splits = min(5, len(unique_groups))
    group_kfold = GroupKFold(n_splits=n_splits)
    fold_scores: list[float] = []

    for train_idx, test_idx in group_kfold.split(
        x,
        y,
        groups,
    ):
        fold_scaler = StandardScaler()
        x_train = fold_scaler.fit_transform(
            x[train_idx]
        )
        x_test = fold_scaler.transform(
            x[test_idx]
        )

        fold_model = make_model(
            n_estimators=200
        )
        fold_model.fit(
            x_train,
            y[train_idx],
        )

        predictions = fold_model.predict(x_test)

        fold_scores.append(
            accuracy_score(
                y[test_idx],
                predictions,
            )
        )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = make_model(
        n_estimators=250
    )
    model.fit(x_scaled, y)

    print("")
    print("RandomForest check")
    print(
        "  Group CV accuracy: "
        f"{np.mean(fold_scores):.3f} "
        f"+/- {np.std(fold_scores):.3f}"
    )

    return model, scaler, x_scaled, y


def get_shap_values(
    model: RandomForestClassifier,
    x_scaled: np.ndarray,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(x_scaled)

    if isinstance(raw_values, list):
        shap_values = np.asarray(
            raw_values,
            dtype=float,
        )
    else:
        raw_array = np.asarray(
            raw_values,
            dtype=float,
        )

        if raw_array.ndim != 3:
            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{raw_array.shape}"
            )

        if raw_array.shape[0] == x_scaled.shape[0]:
            shap_values = np.moveaxis(
                raw_array,
                -1,
                0,
            )
        elif raw_array.shape[0] == len(STATES):
            shap_values = raw_array
        else:
            raise ValueError(
                "Unable to interpret SHAP output shape: "
                f"{raw_array.shape}"
            )

    expected_shape = (
        len(STATES),
        x_scaled.shape[0],
        len(FEATURES),
    )

    if shap_values.shape != expected_shape:
        raise ValueError(
            "Unexpected SHAP values shape: "
            f"{shap_values.shape}; "
            f"expected {expected_shape}."
        )

    return shap_values


def global_importance_table(
    shap_values: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for class_id, state in enumerate(STATES):
        class_importance = np.abs(
            shap_values[class_id]
        ).mean(axis=0)

        for feature, importance in zip(
            FEATURES,
            class_importance,
        ):
            rows.append({
                "state": state,
                "feature": feature,
                "importance": float(importance),
                "concept": FEATURE_TO_CONCEPT[feature],
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "shap_global_by_state.csv",
        ),
        index=False,
    )

    return result


def plot_global_importance(
    global_df: pd.DataFrame,
) -> None:
    summary = (
        global_df
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        summary.index,
        summary.values,
    )
    ax.set_xlabel(
        "Mean absolute SHAP importance"
    )
    ax.set_title(
        "Global feature importance"
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "1_global_feature_importance.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def compare_with_anova(
    global_df: pd.DataFrame,
) -> pd.DataFrame:
    shap_rank = (
        global_df
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    shap_rank["shap_rank"] = np.arange(
        1,
        len(shap_rank) + 1,
    )

    if not os.path.exists(RANKING_FILE):
        output_path = os.path.join(
            OUTPUT_DIR,
            "shap_feature_ranking.csv",
        )

        shap_rank.to_csv(
            output_path,
            index=False,
        )

        print("")
        print(
            "ANOVA ranking file not found; "
            "comparison was skipped."
        )

        return shap_rank

    ranking_df = pd.read_csv(
        RANKING_FILE
    )

    required_columns = {
        "feature",
        "anova_f",
    }

    missing_columns = (
        required_columns.difference(
            ranking_df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in "
            "feature_ranking_per_sport.csv: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    anova_rank = (
        ranking_df
        .dropna(subset=["anova_f"])
        .groupby("feature")["anova_f"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    anova_rank["anova_rank"] = np.arange(
        1,
        len(anova_rank) + 1,
    )

    merged = shap_rank.merge(
        anova_rank,
        on="feature",
        how="left",
    )

    merged["rank_gap"] = (
        merged["shap_rank"]
        - merged["anova_rank"]
    ).abs()

    merged.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "shap_vs_anova_ranking.csv",
        ),
        index=False,
    )

    plot_data = merged.dropna(
        subset=[
            "anova_rank",
            "shap_rank",
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        plot_data["anova_rank"],
        plot_data["shap_rank"],
        s=70,
    )

    for _, row in plot_data.iterrows():
        ax.annotate(
            str(row["feature"]),
            (
                float(row["anova_rank"]),
                float(row["shap_rank"]),
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("ANOVA rank")
    ax.set_ylabel("SHAP rank")
    ax.set_title(
        "SHAP ranking compared with ANOVA ranking"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "2_shap_vs_anova.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return merged


def build_knowledge_graph() -> nx.DiGraph:
    graph = nx.DiGraph()

    for state in STATES:
        graph.add_node(
            state,
            node_type="state",
        )

    concepts = [
        "Heart rate",
        "Breathing rate",
        "RR interval",
    ]

    for concept in concepts:
        graph.add_node(
            concept,
            node_type="concept",
        )

    for feature in FEATURES:
        concept = FEATURE_TO_CONCEPT[feature]

        graph.add_node(
            feature,
            node_type="feature",
        )

        graph.add_edge(
            feature,
            concept,
            relation="belongs to",
        )

    semantic_relations = [
        (
            "Heart rate",
            "S0_Rest",
            "low values support",
        ),
        (
            "Heart rate",
            "S1_Effort",
            "high values support",
        ),
        (
            "Heart rate",
            "S2_Recovery",
            "intermediate values support",
        ),
        (
            "Breathing rate",
            "S0_Rest",
            "low values support",
        ),
        (
            "Breathing rate",
            "S1_Effort",
            "high values support",
        ),
        (
            "Breathing rate",
            "S2_Recovery",
            "intermediate values support",
        ),
        (
            "RR interval",
            "S0_Rest",
            "high values support",
        ),
        (
            "RR interval",
            "S1_Effort",
            "low values support",
        ),
        (
            "RR interval",
            "S2_Recovery",
            "intermediate values support",
        ),
    ]

    for source, target, relation in semantic_relations:
        graph.add_edge(
            source,
            target,
            relation=relation,
        )

    return graph


def plot_knowledge_graph(graph: nx.DiGraph) -> None:
    positions = {
        # Heart-rate features
        "mean_HR": (-4.0, 3.0),
        "std_HR": (-4.0, 2.5),
        "max_HR": (-4.0, 2.0),

        # Breathing-rate features
        "mean_BR": (-4.0, 0.8),
        "std_BR": (-4.0, 0.3),
        "max_BR": (-4.0, -0.2),

        # RR features
        "mean_RR": (-4.0, -1.4),
        "std_RR": (-4.0, -1.9),
        "min_RR": (-4.0, -2.4),

        # Physiological concepts
        "Heart rate": (0.0, 2.5),
        "Breathing rate": (0.0, 0.3),
        "RR interval": (0.0, -1.9),

        # Functional states
        "S0_Rest": (4.0, 2.2),
        "S1_Effort": (4.0, 0.3),
        "S2_Recovery": (4.0, -1.6),
    }

    feature_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes["node_type"] == "feature"
    ]

    concept_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes["node_type"] == "concept"
    ]

    state_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes["node_type"] == "state"
    ]

    feature_edges = [
        (source, target)
        for source, target in graph.edges
        if graph.nodes[source]["node_type"] == "feature"
    ]

    semantic_edges = [
        (source, target)
        for source, target in graph.edges
        if graph.nodes[source]["node_type"] == "concept"
    ]

    fig, ax = plt.subplots(figsize=(15, 8))

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=feature_nodes,
        node_color="#4C9ED9",
        node_size=1500,
        edgecolors="white",
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=concept_nodes,
        node_color="#F39C3D",
        node_size=1900,
        edgecolors="white",
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=state_nodes,
        node_color="#45C97A",
        node_size=1900,
        edgecolors="white",
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=feature_edges,
        edge_color="#A7B0B7",
        width=1.3,
        alpha=0.8,
        arrows=True,
        arrowsize=12,
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=semantic_edges,
        edge_color="#555555",
        width=1.7,
        alpha=0.8,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.06",
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    legend_handles = [
        mpatches.Patch(
            color="#4C9ED9",
            label="Statistical feature",
        ),
        mpatches.Patch(
            color="#F39C3D",
            label="Physiological concept",
        ),
        mpatches.Patch(
            color="#45C97A",
            label="Functional state",
        ),
        Line2D(
            [0],
            [0],
            color="#A7B0B7",
            linewidth=1.5,
            label="Feature belongs to concept",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=1.8,
            label="Concept supports state interpretation",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=5,
        frameon=False,
        fontsize=8,
    )

    ax.text(
        -4.0,
        3.55,
        "Statistical features",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        0.0,
        3.55,
        "Physiological concepts",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        4.0,
        3.55,
        "Functional states",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title(
        "Semantic Knowledge Graph for physiological-state explanation",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-3.1, 4.0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "3_knowledge_graph.png",
        ),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_state_profiles(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    sport_profiles = (
        df
        .groupby(
            ["sport", "true_state"]
        )[FEATURES]
        .median()
    )

    global_profiles = (
        df
        .groupby(
            "true_state"
        )[FEATURES]
        .median()
        .reindex(STATES)
    )

    sport_scales = (
        df
        .groupby(
            "sport"
        )[FEATURES]
        .std()
        .replace(0, np.nan)
    )

    global_scales = (
        df[FEATURES]
        .std()
        .replace(0, np.nan)
        .fillna(1.0)
    )

    sport_profiles.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "state_profiles_by_sport.csv",
        )
    )

    global_profiles.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "state_profiles_global.csv",
        )
    )

    return (
        sport_profiles,
        global_profiles,
        sport_scales,
        global_scales,
    )


def get_available_profiles(
    sport: str,
    sport_profiles: pd.DataFrame,
    global_profiles: pd.DataFrame,
) -> pd.DataFrame:
    sport_level = (
        sport_profiles
        .index
        .get_level_values(0)
        .unique()
    )

    if sport in sport_level:
        profiles = sport_profiles.loc[sport]

        if set(STATES).issubset(
            set(profiles.index)
        ):
            return profiles.reindex(STATES)

    return global_profiles.reindex(STATES)


def get_feature_scale(
    sport: str,
    feature: str,
    sport_scales: pd.DataFrame,
    global_scales: pd.Series,
) -> float:
    if sport in sport_scales.index:
        scale = sport_scales.loc[
            sport,
            feature,
        ]

        if pd.notna(scale) and float(scale) > 0:
            return float(scale)

    fallback_scale = global_scales.get(
        feature,
        1.0,
    )

    if (
        pd.isna(fallback_scale)
        or float(fallback_scale) <= 0
    ):
        return 1.0

    return float(fallback_scale)


def feature_state_similarity(
    row: pd.Series,
    predicted_state: str,
    feature: str,
    sport_profiles: pd.DataFrame,
    global_profiles: pd.DataFrame,
    sport_scales: pd.DataFrame,
    global_scales: pd.Series,
) -> float:
    sport = str(row["sport"])
    value = float(row[feature])

    profiles = get_available_profiles(
        sport,
        sport_profiles,
        global_profiles,
    )

    if predicted_state not in profiles.index:
        return 0.0

    scale = get_feature_scale(
        sport,
        feature,
        sport_scales,
        global_scales,
    )

    distances: list[float] = []

    for state in STATES:
        reference_value = profiles.loc[
            state,
            feature,
        ]

        if pd.isna(reference_value):
            distances.append(np.inf)
        else:
            distances.append(
                abs(
                    value
                    - float(reference_value)
                )
                / scale
            )

    distance_array = np.asarray(
        distances,
        dtype=float,
    )

    finite_mask = np.isfinite(
        distance_array
    )

    if not finite_mask.any():
        return 0.0

    similarities = np.zeros_like(
        distance_array
    )

    finite_distances = distance_array[
        finite_mask
    ]

    similarities[finite_mask] = np.exp(
        -(
            finite_distances
            - finite_distances.min()
        )
    )

    total = float(similarities.sum())

    if total <= 0:
        return 0.0

    state_index = STATES.index(
        predicted_state
    )

    return float(
        similarities[state_index]
        / total
    )


def select_supporting_features(
    class_shap_values: np.ndarray,
    max_features: int = 3,
) -> list[int]:
    ordered_indices = np.argsort(
        class_shap_values
    )[::-1]

    supporting_indices: list[int] = []

    for index in ordered_indices:
        if class_shap_values[index] > 0:
            supporting_indices.append(
                int(index)
            )

        if len(supporting_indices) >= max_features:
            break

    return supporting_indices


def kg_consistency(
    row: pd.Series,
    predicted_state: str,
    class_shap_values: np.ndarray,
    sport_profiles: pd.DataFrame,
    global_profiles: pd.DataFrame,
    sport_scales: pd.DataFrame,
    global_scales: pd.Series,
    max_features: int = 5,
) -> tuple[float, list[dict[str, Any]]]:
    supporting_indices = select_supporting_features(
        class_shap_values,
        max_features=max_features,
    )

    if not supporting_indices:
        return 0.0, []

    similarities: list[float] = []
    weights: list[float] = []
    details: list[dict[str, Any]] = []

    for feature_index in supporting_indices:
        feature = FEATURES[feature_index]
        shap_value = float(
            class_shap_values[feature_index]
        )

        similarity = feature_state_similarity(
            row=row,
            predicted_state=predicted_state,
            feature=feature,
            sport_profiles=sport_profiles,
            global_profiles=global_profiles,
            sport_scales=sport_scales,
            global_scales=global_scales,
        )

        similarities.append(similarity)
        weights.append(abs(shap_value))

        details.append({
            "feature": feature,
            "value": float(row[feature]),
            "shap_value": shap_value,
            "similarity": similarity,
            "concept": FEATURE_TO_CONCEPT[feature],
        })

    weight_sum = float(sum(weights))

    if weight_sum <= 0:
        return 0.0, details

    score = np.average(
        np.asarray(
            similarities,
            dtype=float,
        ),
        weights=np.asarray(
            weights,
            dtype=float,
        ),
    )

    return float(score), details


def make_text_explanation(
    predicted_state: str,
    confidence: float,
    supporting_features: list[dict[str, Any]],
    consistency: float,
) -> str:
    parts: list[str] = []

    for item in supporting_features:
        feature = str(
            item.get(
                "feature",
                "",
            )
        )

        if not feature:
            continue

        value = float(item["value"])
        similarity = float(
            item["similarity"]
        )
        concept = str(
            item["concept"]
        )

        parts.append(
            f"{feature}={value:.1f} "
            f"({concept}, "
            f"semantic match={similarity:.2f})"
        )

    state_name = predicted_state.replace(
        "_",
        " ",
    )

    if parts:
        feature_text = ", ".join(parts)
    else:
        feature_text = (
            "no positive SHAP contribution "
            "was available"
        )

    return (
        f"The model predicts {state_name} "
        f"with {confidence:.1%} confidence. "
        f"The main supporting features are "
        f"{feature_text}. "
        f"The semantic consistency score is "
        f"{consistency:.2f}."
    )


def hitl_flag(
    confidence: float,
    consistency: float,
) -> str:
    if (
        confidence >= 0.75
        and consistency >= 0.70
    ):
        return "accept"

    if (
        confidence >= 0.55
        and consistency >= 0.45
    ):
        return "check"

    return "review"


def local_explanations(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    scaler: StandardScaler,
    shap_values: np.ndarray,
    sport_profiles: pd.DataFrame,
    global_profiles: pd.DataFrame,
    sport_scales: pd.DataFrame,
    global_scales: pd.Series,
) -> pd.DataFrame:
    x = scaler.transform(
        df[FEATURES].to_numpy(dtype=float)
    )

    predictions = model.predict(x)
    probabilities = model.predict_proba(x)

    rows: list[dict[str, Any]] = []

    for index, row in df.iterrows():
        predicted_id = int(
            predictions[index]
        )

        predicted_state = ID_TO_STATE[
            predicted_id
        ]

        confidence = float(
            probabilities[
                index,
                predicted_id,
            ]
        )

        class_shap_values = np.asarray(
            shap_values[
                predicted_id,
                index,
            ],
            dtype=float,
        )

        consistency, consistency_details = kg_consistency(
            row=row,
            predicted_state=predicted_state,
            class_shap_values=class_shap_values,
            sport_profiles=sport_profiles,
            global_profiles=global_profiles,
            sport_scales=sport_scales,
            global_scales=global_scales,
        )

        supporting_features = [
            dict(item)
            for item in consistency_details[:3]
        ]

        while len(supporting_features) < 3:
            supporting_features.append({
                "feature": "",
                "value": np.nan,
                "shap_value": np.nan,
                "similarity": np.nan,
                "concept": "",
            })

        explanation = make_text_explanation(
            predicted_state=predicted_state,
            confidence=confidence,
            supporting_features=supporting_features,
            consistency=consistency,
        )

        rows.append({
            "window_id": int(
                row["window_id"]
            ),
            "sport": str(
                row["sport"]
            ),
            "subject": str(
                row["subject"]
            ),
            "true_state": str(
                row["true_state"]
            ),
            "prediction": predicted_state,
            "confidence": round(
                confidence,
                3,
            ),
            "top_feature_1": supporting_features[0]["feature"],
            "top_feature_2": supporting_features[1]["feature"],
            "top_feature_3": supporting_features[2]["feature"],
            "top_value_1": supporting_features[0]["value"],
            "top_value_2": supporting_features[1]["value"],
            "top_value_3": supporting_features[2]["value"],
            "top_importance_1": supporting_features[0]["shap_value"],
            "top_importance_2": supporting_features[1]["shap_value"],
            "top_importance_3": supporting_features[2]["shap_value"],
            "top_similarity_1": supporting_features[0]["similarity"],
            "top_similarity_2": supporting_features[1]["similarity"],
            "top_similarity_3": supporting_features[2]["similarity"],
            "kg_consistency": round(
                consistency,
                3,
            ),
            "hitl_flag": hitl_flag(
                confidence,
                consistency,
            ),
            "explanation": explanation,
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "local_explanations.csv",
        ),
        index=False,
    )

    return result


def plot_consistency(
    local_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    counts = (
        local_df["hitl_flag"]
        .value_counts()
        .reindex(
            [
                "accept",
                "check",
                "review",
            ],
            fill_value=0,
        )
    )

    axes[0].bar(
        counts.index,
        counts.values,
    )
    axes[0].set_title(
        "Human-in-the-loop flags"
    )
    axes[0].set_ylabel(
        "Number of windows"
    )

    state_consistency = (
        local_df
        .groupby(
            "prediction"
        )["kg_consistency"]
        .mean()
        .reindex(STATES)
    )

    state_labels = [
        state.replace(
            "_",
            "\n",
        )
        for state in state_consistency.index
    ]

    axes[1].bar(
        state_labels,
        state_consistency.values,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel(
        "Mean semantic consistency"
    )
    axes[1].set_title(
        "Semantic consistency by predicted state"
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "4_consistency_and_hitl.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_examples(
    local_df: pd.DataFrame,
) -> None:
    print("")
    print("Example explanations")

    examples = (
        local_df
        .sort_values(
            [
                "kg_consistency",
                "confidence",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(5)
    )

    for _, row in examples.iterrows():
        print("")
        print(
            f"  sport={row['sport']} | "
            f"true={row['true_state']} | "
            f"pred={row['prediction']}"
        )
        print(
            f"  confidence={row['confidence']} | "
            f"consistency={row['kg_consistency']}"
        )
        print(
            f"  {row['explanation']}"
        )


def print_consistency_summary(
    local_df: pd.DataFrame,
) -> None:
    print("")
    print(
        "Mean semantic consistency "
        "by predicted state"
    )

    state_summary = (
        local_df
        .groupby(
            "prediction"
        )["kg_consistency"]
        .agg([
            "mean",
            "std",
            "count",
        ])
        .reindex(STATES)
        .round(3)
    )

    print(
        state_summary.to_string()
    )

    print("")
    print("HITL distribution")

    hitl_summary = (
        local_df["hitl_flag"]
        .value_counts(
            normalize=True
        )
        .reindex(
            [
                "accept",
                "check",
                "review",
            ],
            fill_value=0,
        )
        .mul(100)
        .round(1)
        .astype(str)
        .add("%")
    )

    print(
        hitl_summary.to_string()
    )


def main() -> None:
    df = load_data()

    (
        sport_profiles,
        global_profiles,
        sport_scales,
        global_scales,
    ) = build_state_profiles(df)

    (
        model,
        scaler,
        x_scaled,
        _,
    ) = train_random_forest(df)

    shap_values = get_shap_values(
        model,
        x_scaled,
    )

    print("")
    print("Explanation method: SHAP")

    global_df = global_importance_table(
        shap_values
    )

    ranking_df = compare_with_anova(
        global_df
    )

    graph = build_knowledge_graph()

    plot_global_importance(
        global_df
    )

    plot_knowledge_graph(
        graph
    )

    local_df = local_explanations(
        df=df,
        model=model,
        scaler=scaler,
        shap_values=shap_values,
        sport_profiles=sport_profiles,
        global_profiles=global_profiles,
        sport_scales=sport_scales,
        global_scales=global_scales,
    )

    plot_consistency(
        local_df
    )

    print("")
    print("Top global features")

    top_features = (
        global_df
        .groupby(
            "feature"
        )["importance"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(6)
        .round(4)
    )

    print(
        top_features.to_string()
    )

    print("")
    print("SHAP / ANOVA ranking")

    print(
        ranking_df
        .round(3)
        .to_string(index=False)
    )

    print_consistency_summary(
        local_df
    )

    print_examples(
        local_df
    )

    print("")
    print(
        f"Generated files in: "
        f"{OUTPUT_DIR}"
    )
    print(
        "  1_global_feature_importance.png"
    )
    print(
        "  2_shap_vs_anova.png"
    )
    print(
        "  3_knowledge_graph.png"
    )
    print(
        "  4_consistency_and_hitl.png"
    )
    print(
        "  shap_global_by_state.csv"
    )
    print(
        "  shap_vs_anova_ranking.csv"
    )
    print(
        "  local_explanations.csv"
    )
    print(
        "  state_profiles_by_sport.csv"
    )
    print(
        "  state_profiles_global.csv"
    )
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
