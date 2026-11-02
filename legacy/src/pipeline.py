import os
import re
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import _tree

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUN_ROOT = os.path.join(BASE_DIR, "SportDB2", "RUN")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 20
STRIDE = 10
RANDOM_SEED = 42
MAX_RULES_PER_CLASS = 10

LABEL_NAMES = [
    "S0_Resting",
    "S1_Effort_Initiation",
    "S2_Intense_Effort",
    "S3_Sustained_Effort",
    "S4_Peak_Load",
    "S5_Recovery",
]

STATE_LABELS = {name: i for i, name in enumerate(LABEL_NAMES)}

FEATURE_NAMES = [
    "mean_HR", "std_HR", "max_HR",
    "mean_BR", "std_BR", "max_BR",
    "mean_RR", "std_RR", "min_RR",
]

FEATURE_TO_NODE = {
    "mean_HR": "Heart Rate (HR)",
    "std_HR": "Heart Rate (HR)",
    "max_HR": "Heart Rate (HR)",
    "mean_BR": "Breathing Rate (BR)",
    "std_BR": "Breathing Rate (BR)",
    "max_BR": "Breathing Rate (BR)",
    "mean_RR": "RR Interval",
    "std_RR": "RR Interval",
    "min_RR": "RR Interval",
}

UNIT_MAP = {
    "mean_HR": "bpm",
    "std_HR": "bpm",
    "max_HR": "bpm",
    "mean_BR": "rpm",
    "std_BR": "rpm",
    "max_BR": "rpm",
    "mean_RR": "ms",
    "std_RR": "ms",
    "min_RR": "ms",
}


def time_to_seconds(time_string):
    h, m, s = map(int, time_string.split(":"))
    return h * 3600 + m * 60 + s


def parse_trnote(path):
    with open(path, "r", encoding="latin-1") as file:
        text = file.read()

    patterns = [
        ("S0_Resting", r"Resting\s*:\s*from\s*(\d\d:\d\d:\d\d)\s*to\s*(\d\d:\d\d:\d\d)"),
        ("S1_Effort_Initiation", r"I Phase\s*:\s*from\s*(\d\d:\d\d:\d\d)\s*to\s*(\d\d:\d\d:\d\d)"),
        ("S2_Intense_Effort", r"II Phase\s*:\s*from\s*(\d\d:\d\d:\d\d)\s*to\s*(\d\d:\d\d:\d\d)"),
        ("S3_Sustained_Effort", r"III Phase\s*:\s*from\s*(\d\d:\d\d:\d\d)\s*to\s*(\d\d:\d\d:\d\d)"),
        ("S4_Peak_Load", r"IV Phase\s*:\s*from\s*(\d\d:\d\d:\d\d)\s*to\s*(\d\d:\d\d:\d\d)"),
        ("S5_Recovery", r"Recovery\s*:\s*from\s*(\d\d:\d\d:\d\d)\s*to\s*(\d\d:\d\d:\d\d)"),
    ]

    segments = {}

    for state, pattern in patterns:
        match = re.search(pattern, text)
        if match:
            segments[state] = (
                time_to_seconds(match.group(1)),
                time_to_seconds(match.group(2)),
            )

    return segments


def extract_window_features(hr, br, rr):
    hr = hr[np.isfinite(hr)]
    br = br[np.isfinite(br)]
    rr = rr[np.isfinite(rr)]

    if len(hr) == 0 or len(br) == 0 or len(rr) == 0:
        return [np.nan] * len(FEATURE_NAMES)

    return [
        np.mean(hr), np.std(hr), np.max(hr),
        np.mean(br), np.std(br), np.max(br),
        np.mean(rr), np.std(rr), np.min(rr),
    ]


def load_run_dataset(root):
    X_list = []
    y_list = []
    groups = []
    meta = []
    used_files = []
    skipped_files = []

    for subject in sorted(os.listdir(root)):
        subject_path = os.path.join(root, subject)

        if not os.path.isdir(subject_path):
            continue

        for crd in sorted(os.listdir(subject_path)):
            crd_path = os.path.join(subject_path, crd)
            data_path = os.path.join(crd_path, "Data.mat")
            trnote_path = os.path.join(crd_path, "TrNote.txt")

            if not os.path.exists(data_path) or not os.path.exists(trnote_path):
                continue

            try:
                data = loadmat(data_path)
                D = data["Data"]

                if not all(field in D.dtype.names for field in ["HR", "BR", "RR"]):
                    skipped_files.append((subject, crd, "missing HR/BR/RR"))
                    continue

                HR = D[0, 0]["HR"].flatten().astype(float)
                BR = D[0, 0]["BR"].flatten().astype(float)
                RR = D[0, 0]["RR"].flatten().astype(float)

                segments = parse_trnote(trnote_path)

                if len(segments) < 6:
                    skipped_files.append((subject, crd, "incomplete protocol"))
                    continue

                total_seconds = max(end for _, end in segments.values())
                samples_per_second = len(HR) / total_seconds

                for state_name, (start_sec, end_sec) in segments.items():
                    start = int(start_sec * samples_per_second)
                    end = int(end_sec * samples_per_second)

                    if end - start < WINDOW_SIZE:
                        continue

                    for i in range(start, end - WINDOW_SIZE, STRIDE):
                        j = i + WINDOW_SIZE
                        features = extract_window_features(HR[i:j], BR[i:j], RR[i:j])

                        X_list.append(features)
                        y_list.append(STATE_LABELS[state_name])
                        groups.append(subject)
                        meta.append({
                            "subject": subject,
                            "crd": crd,
                            "state": state_name,
                            "window_start": i,
                            "window_end": j,
                        })

                used_files.append((subject, crd))

            except Exception as error:
                skipped_files.append((subject, crd, str(error)))

    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(groups)

    valid_mask = ~np.isnan(X).any(axis=1)

    X = X[valid_mask]
    y = y[valid_mask]
    groups = groups[valid_mask]
    meta = [m for m, keep in zip(meta, valid_mask) if keep]

    return X, y, groups, meta, used_files, skipped_files


def build_random_forest(n_estimators=200):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )


def evaluate_loso(X, y, groups):
    scores = []
    subjects = sorted(set(groups))

    for subject in subjects:
        test_mask = groups == subject
        train_mask = ~test_mask

        model = build_random_forest(n_estimators=150)
        model.fit(X[train_mask], y[train_mask])

        scores.append(model.score(X[test_mask], y[test_mask]))

    return np.array(scores), subjects


def compute_importance(model, X, y):
    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        importance_matrix = np.array([
            np.abs(values).mean(axis=0)
            for values in shap_values
        ])

        method = "SHAP"

    else:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=15,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        importance_matrix = np.tile(
            result.importances_mean,
            (len(LABEL_NAMES), 1),
        )

        shap_values = [
            np.tile(importance_matrix[class_id], (X.shape[0], 1))
            for class_id in range(len(LABEL_NAMES))
        ]

        method = "Permutation importance"

    return shap_values, importance_matrix, method


def extract_rules_from_forest(forest, feature_names, class_names):
    rules = []

    for estimator in forest.estimators_[:20]:
        tree = estimator.tree_

        def walk(node, path):
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                samples = tree.n_node_samples[node]

                if samples < 5:
                    return

                values = tree.value[node][0]
                class_id = int(np.argmax(values))
                confidence = values[class_id] / values.sum()

                if confidence >= 0.6:
                    rules.append({
                        "conditions": list(path),
                        "class_name": class_names[class_id],
                        "support": int(samples),
                        "confidence": float(confidence),
                    })

                return

            feature = feature_names[tree.feature[node]]
            threshold = round(tree.threshold[node], 2)

            walk(tree.children_left[node], path + [(feature, "<=", threshold)])
            walk(tree.children_right[node], path + [(feature, ">", threshold)])

        walk(0, [])

    rules = sorted(
        rules,
        key=lambda rule: rule["support"] * rule["confidence"],
        reverse=True,
    )

    selected_rules = []
    counter = Counter()

    for rule in rules:
        class_name = rule["class_name"]

        if counter[class_name] < MAX_RULES_PER_CLASS:
            selected_rules.append(rule)
            counter[class_name] += 1

    return selected_rules


def apply_rules(feature_dict, rules):
    best_class = None
    best_confidence = 0.0
    best_rule = ""

    for rule in rules:
        match = True

        for feature, operator, threshold in rule["conditions"][:4]:
            value = feature_dict.get(feature)

            if value is None:
                match = False
                break

            if operator == "<=" and value > threshold:
                match = False
                break

            if operator == ">" and value <= threshold:
                match = False
                break

        if match and rule["confidence"] > best_confidence:
            best_class = rule["class_name"]
            best_confidence = rule["confidence"]
            best_rule = format_rule(rule)

    return best_class, best_confidence, best_rule


def format_rule(rule, max_conditions=4):
    conditions = []

    for feature, operator, threshold in rule["conditions"][:max_conditions]:
        conditions.append(f"{feature} {operator} {threshold}")

    return " AND ".join(conditions)


def consistency_score(model_label, rule_label):
    if model_label is None or rule_label is None:
        return 0.0

    if model_label == rule_label:
        return 1.0

    distance = abs(LABEL_NAMES.index(model_label) - LABEL_NAMES.index(rule_label))

    if distance == 1:
        return 0.7

    if distance == 2:
        return 0.4

    return 0.0


def build_knowledge_graph(X, y, importance_matrix, importance_method):
    G = nx.DiGraph()

    for state in LABEL_NAMES:
        G.add_node(state, node_type="fss_state")

    for node in ["Heart Rate (HR)", "Breathing Rate (BR)", "RR Interval"]:
        G.add_node(node, node_type="physiological")

    for concept in [
        "Cardiorespiratory\nResponse",
        "Physiological\nLoad",
        "Fatigue Risk",
        "Recovery Process",
    ]:
        G.add_node(concept, node_type="semantic")

    for source, target in zip(LABEL_NAMES[:-1], LABEL_NAMES[1:]):
        G.add_edge(source, target, relation="transition_to", color="#3498DB")

    semantic_edges = [
        ("Heart Rate (HR)", "Cardiorespiratory\nResponse", "indicates"),
        ("Breathing Rate (BR)", "Cardiorespiratory\nResponse", "indicates"),
        ("RR Interval", "Cardiorespiratory\nResponse", "indicates"),
        ("Cardiorespiratory\nResponse", "Physiological\nLoad", "reflects"),
        ("Physiological\nLoad", "Fatigue Risk", "may_trigger"),
        ("Recovery Process", "Fatigue Risk", "reduces"),
    ]

    for source, target, relation in semantic_edges:
        G.add_edge(source, target, relation=relation, color="#E74C3C")

    df_features = pd.DataFrame(X, columns=FEATURE_NAMES)
    df_features["state"] = [LABEL_NAMES[i] for i in y]

    for class_id, state in enumerate(LABEL_NAMES):
        grouped_importance = {}

        for feature_id, feature_name in enumerate(FEATURE_NAMES):
            node_name = FEATURE_TO_NODE[feature_name]
            grouped_importance[node_name] = (
                grouped_importance.get(node_name, 0.0)
                + importance_matrix[class_id, feature_id]
            )

        best_node = max(grouped_importance, key=grouped_importance.get)
        best_feature = FEATURE_NAMES[np.argmax(importance_matrix[class_id])]

        state_data = df_features[df_features["state"] == state]
        mean_value = state_data[best_feature].mean()
        unit = UNIT_MAP.get(best_feature, "")

        G.add_edge(
            best_node,
            state,
            relation=f"{best_feature}={mean_value:.0f} {unit}",
            color="#2ECC71",
            source=importance_method,
        )

    G.add_edge("Physiological\nLoad", "S4_Peak_Load", relation="characterizes", color="#9B59B6")
    G.add_edge("Physiological\nLoad", "S2_Intense_Effort", relation="characterizes", color="#9B59B6")
    G.add_edge("Recovery Process", "S5_Recovery", relation="characterizes", color="#9B59B6")
    G.add_edge("Cardiorespiratory\nResponse", "S0_Resting", relation="baseline", color="#9B59B6")

    return G


def build_results_table(X, y, groups, meta, model, rules, shap_values):
    rows = []

    for idx, features in enumerate(X):
        feature_dict = dict(zip(FEATURE_NAMES, features))

        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        predicted_id = int(np.argmax(probabilities))
        model_label = LABEL_NAMES[predicted_id]
        model_confidence = probabilities[predicted_id]

        rule_label, rule_confidence, rule_text = apply_rules(feature_dict, rules)
        score = consistency_score(model_label, rule_label)

        shap_window = np.array([
            shap_values[class_id][idx]
            for class_id in range(len(LABEL_NAMES))
        ])

        top_feature_id = int(np.argmax(np.abs(shap_window[predicted_id])))
        top_feature = FEATURE_NAMES[top_feature_id]
        top_importance = float(shap_window[predicted_id, top_feature_id])

        explanation = (
            f"The model predicted {model_label.replace('_', ' ')} "
            f"with {model_confidence:.1%} confidence. "
            f"The main contributing feature was {top_feature} "
            f"with value {feature_dict[top_feature]:.1f}. "
            f"The rule-based inference was {(rule_label or 'None').replace('_', ' ')}. "
            f"The consistency score was {score:.2f}."
        )

        rows.append({
            "window_id": idx,
            "subject": groups[idx],
            "crd": meta[idx]["crd"],
            "true_state": LABEL_NAMES[y[idx]],
            "model_prediction": model_label,
            "model_confidence": round(model_confidence, 3),
            "rule_prediction": rule_label or "None",
            "rule_confidence": round(rule_confidence, 3),
            "rule_used": rule_text or "None",
            "consistency": score,
            "top_feature": top_feature,
            "top_feature_value": round(float(feature_dict[top_feature]), 3),
            "top_importance": round(top_importance, 4),
            "explanation": explanation,
        })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(f"{OUTPUT_DIR}/results_explainable.csv", index=False)

    return df_results


def save_confusion_matrix(y, y_pred):
    fig, ax = plt.subplots(figsize=(9, 7))

    ConfusionMatrixDisplay(
        confusion_matrix(y, y_pred),
        display_labels=[state.replace("_", "\n") for state in LABEL_NAMES],
    ).plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title("Confusion Matrix - Random Forest Multi-subject")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_confusion_matrix.png", dpi=150)
    plt.close()


def save_shap_summary(shap_values, X):
    if not SHAP_AVAILABLE:
        return

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X,
        feature_names=FEATURE_NAMES,
        class_names=[state.replace("_", " ") for state in LABEL_NAMES],
        show=False,
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_knowledge_graph(G, importance_method):
    position = {
        "S0_Resting": (0, 5),
        "S1_Effort_Initiation": (0, 4),
        "S2_Intense_Effort": (0, 3),
        "S3_Sustained_Effort": (0, 2),
        "S4_Peak_Load": (0, 1),
        "S5_Recovery": (0, 0),
        "Heart Rate (HR)": (-3.5, 3.5),
        "Breathing Rate (BR)": (-3.5, 2.0),
        "RR Interval": (-3.5, 0.5),
        "Cardiorespiratory\nResponse": (3.5, 4.0),
        "Physiological\nLoad": (3.5, 2.5),
        "Fatigue Risk": (3.5, 1.0),
        "Recovery Process": (3.5, -0.5),
    }

    node_color_map = {
        "fss_state": "#3498DB",
        "physiological": "#E74C3C",
        "semantic": "#2ECC71",
    }

    fig, ax = plt.subplots(figsize=(18, 11))

    nx.draw_networkx_nodes(
        G,
        position,
        node_color=[node_color_map[G.nodes[node]["node_type"]] for node in G.nodes],
        node_size=2600,
        alpha=0.92,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        position,
        font_size=7.5,
        font_weight="bold",
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        position,
        edge_color=[G.edges[edge].get("color", "#555555") for edge in G.edges],
        arrows=True,
        arrowsize=14,
        width=1.5,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    nx.draw_networkx_edge_labels(
        G,
        position,
        edge_labels=nx.get_edge_attributes(G, "relation"),
        font_size=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
        ax=ax,
    )

    ax.legend(
        handles=[
            mpatches.Patch(color="#3498DB", label="FSS states"),
            mpatches.Patch(color="#E74C3C", label="Physiological indicators"),
            mpatches.Patch(color="#2ECC71", label="Semantic concepts"),
            plt.Line2D([0], [0], color="#2ECC71", lw=2, label=f"Feature-state link ({importance_method})"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )

    ax.set_title(f"Knowledge Graph based on {importance_method}", fontsize=12, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_knowledge_graph.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_importance(model, global_importance, importance_method):
    mdi_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    explanation_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": global_importance,
    }).sort_values("importance", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(mdi_df["feature"], mdi_df["importance"])
    axes[0].set_title("Feature importance - MDI")
    axes[0].invert_yaxis()

    axes[1].barh(explanation_df["feature"], explanation_df["importance"])
    axes[1].set_title(f"Feature importance - mean absolute {importance_method}")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_loso_and_consistency(loso_scores, state_consistency, mean_consistency):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].bar(range(len(loso_scores)), loso_scores)
    axes[0].axhline(
        loso_scores.mean(),
        color="black",
        linestyle="--",
        label=f"Mean={loso_scores.mean():.2f}",
    )

    axes[0].set_title("Leave-one-subject-out accuracy")
    axes[0].set_xlabel("Subject left out")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()

    axes[1].bar(
        [state.replace("_", "\n") for state in LABEL_NAMES],
        state_consistency.values,
    )

    axes[1].axhline(
        mean_consistency,
        color="black",
        linestyle="--",
        label=f"Mean={mean_consistency:.2f}",
    )

    axes[1].set_title("AI-KG consistency by state")
    axes[1].set_ylabel("Mean consistency")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_loso_and_consistency.png", dpi=150)
    plt.close()


def save_consistency_distribution(df_results):
    total_count = len(df_results)

    inconsistent_count = (df_results["consistency"] == 0.0).sum()
    partial_count = df_results["consistency"].between(0.1, 0.99).sum()
    perfect_count = (df_results["consistency"] == 1.0).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    values = [inconsistent_count, partial_count, perfect_count]
    labels = ["Inconsistent", "Partial", "Perfect"]

    axes[0].bar(labels, values)
    axes[0].set_title("AI-KG consistency distribution")
    axes[0].set_ylabel("Number of windows")

    for i, value in enumerate(values):
        percentage = 100 * value / total_count
        axes[0].text(
            i,
            value + 3,
            f"{value}\n({percentage:.1f}%)",
            ha="center",
            fontsize=9,
        )

    feature_by_state = (
        df_results
        .groupby(["true_state", "top_feature"])
        .size()
        .unstack(fill_value=0)
        .reindex(LABEL_NAMES)
    )

    bottom = np.zeros(len(LABEL_NAMES))

    for feature_name in FEATURE_NAMES:
        if feature_name not in feature_by_state.columns:
            continue

        values = feature_by_state[feature_name].values

        axes[1].bar(
            [state.replace("_", "\n") for state in LABEL_NAMES],
            values,
            bottom=bottom,
            label=feature_name,
            alpha=0.85,
        )

        bottom += values

    axes[1].set_title("Top feature by FSS state")
    axes[1].set_ylabel("Number of windows")
    axes[1].legend(fontsize=7, ncol=3, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_consistency_and_top_features.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    X, y, groups, meta, used_files, skipped_files = load_run_dataset(RUN_ROOT)

    model = build_random_forest()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    cv_scores = cross_val_score(model, X, y, cv=cv)
    loso_scores, _ = evaluate_loso(X, y, groups)

    model.fit(X, y)
    y_pred = model.predict(X)

    shap_values, importance_matrix, importance_method = compute_importance(model, X, y)
    global_importance = importance_matrix.mean(axis=0)

    rules = extract_rules_from_forest(model, FEATURE_NAMES, LABEL_NAMES)
    graph = build_knowledge_graph(X, y, importance_matrix, importance_method)

    df_results = build_results_table(
        X,
        y,
        groups,
        meta,
        model,
        rules,
        shap_values,
    )

    mean_consistency = df_results["consistency"].mean()

    state_consistency = (
        df_results
        .groupby("true_state")["consistency"]
        .mean()
        .reindex(LABEL_NAMES)
    )

    save_confusion_matrix(y, y_pred)
    save_shap_summary(shap_values, X)
    save_knowledge_graph(graph, importance_method)
    save_feature_importance(model, global_importance, importance_method)
    save_loso_and_consistency(loso_scores, state_consistency, mean_consistency)
    save_consistency_distribution(df_results)

    rule_count = Counter(rule["class_name"] for rule in rules)
    top_features = [
        FEATURE_NAMES[i]
        for i in np.argsort(global_importance)[::-1][:3]
    ]

    print("Dataset")
    print(f"  Files used: {len(used_files)}")
    print(f"  Subjects: {len(set(groups))}")
    print(f"  Windows: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")

    if skipped_files:
        print(f"  Skipped files: {len(skipped_files)}")

    print("")
    print("Model")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  LOSO accuracy: {loso_scores.mean():.3f} +/- {loso_scores.std():.3f}")
    print(f"  Training accuracy: {(y_pred == y).mean():.3f}")

    print("")
    print("Explainability")
    print(f"  Importance method: {importance_method}")
    print(f"  Top features: {top_features}")
    print(f"  Rules extracted: {len(rules)}")
    print(f"  Rules per class: {dict(rule_count)}")
    print(f"  Mean AI-KG consistency: {mean_consistency:.3f}")

    print("")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()