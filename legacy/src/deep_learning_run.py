import os
import re
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(BASE_DIR, "SportDB2", "RUN")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_dl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
WINDOW_SIZE = 50
STRIDE = 25

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 6

LABEL_NAMES = [
    "S0_Resting",
    "S1_Effort_Initiation",
    "S2_Intense_Effort",
    "S3_Sustained_Effort",
    "S4_Peak_Load",
    "S5_Recovery",
]

STATE_LABELS = {state: i for i, state in enumerate(LABEL_NAMES)}

FEATURE_NAMES = [
    "mean_HR", "std_HR", "max_HR",
    "mean_BR", "std_BR", "max_BR",
    "mean_RR", "std_RR", "min_RR",
]

CHANNEL_NAMES = ["HR", "BR", "RR"]

ALLOWED_TRANSITIONS = {
    0: [0, 1],
    1: [1, 2],
    2: [2, 3, 4],
    3: [3, 4],
    4: [4, 5],
    5: [5],
}

np.random.seed(RANDOM_SEED)

if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)


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


def compute_stat_features(hr, br, rr):
    return [
        np.mean(hr), np.std(hr), np.max(hr),
        np.mean(br), np.std(br), np.max(br),
        np.mean(rr), np.std(rr), np.min(rr),
    ]


def load_run_windows():
    X_seq = []
    X_stat = []
    y = []
    groups = []
    meta = []
    used_files = []
    skipped_files = []

    for subject in sorted(os.listdir(RUN_ROOT)):
        subject_path = os.path.join(RUN_ROOT, subject)

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

                    for i in range(start, end - WINDOW_SIZE, STRIDE):
                        j = i + WINDOW_SIZE

                        hr_window = HR[i:j]
                        br_window = BR[i:j]
                        rr_window = RR[i:j]

                        if (
                            len(hr_window) != WINDOW_SIZE
                            or len(br_window) != WINDOW_SIZE
                            or len(rr_window) != WINDOW_SIZE
                        ):
                            continue

                        if (
                            not np.isfinite(hr_window).all()
                            or not np.isfinite(br_window).all()
                            or not np.isfinite(rr_window).all()
                        ):
                            continue

                        X_seq.append(np.stack([hr_window, br_window, rr_window], axis=1))
                        X_stat.append(compute_stat_features(hr_window, br_window, rr_window))
                        y.append(STATE_LABELS[state_name])
                        groups.append(subject)

                        meta.append({
                            "subject": subject,
                            "crd": crd,
                            "true_state": state_name,
                            "window_start": i,
                            "window_end": j,
                        })

                used_files.append((subject, crd))

            except Exception as error:
                skipped_files.append((subject, crd, str(error)))

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_stat, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(groups),
        meta,
        used_files,
        skipped_files,
    )


def temporal_consistency_score(predictions, groups):
    scores = np.ones(len(predictions), dtype=float)

    for subject in sorted(set(groups)):
        idx = np.where(groups == subject)[0]
        subject_preds = predictions[idx]

        for k in range(1, len(subject_preds)):
            prev_state = int(subject_preds[k - 1])
            current_state = int(subject_preds[k])

            if current_state in ALLOWED_TRANSITIONS.get(prev_state, []):
                scores[idx[k]] = 1.0
            else:
                distance = abs(current_state - prev_state)
                scores[idx[k]] = 0.5 if distance <= 2 else 0.0

    return scores


def hitl_flag(confidence, temporal_score):
    if confidence < 0.45:
        return "review"
    if temporal_score < 0.5:
        return "review"
    if confidence < 0.60 or temporal_score < 1.0:
        return "check"
    return "accept"


def scale_stat_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def scale_sequence_data(X_train, X_test):
    scaler = StandardScaler()

    n_train, seq_len, n_channels = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(-1, n_channels)
    X_test_flat = X_test.reshape(-1, n_channels)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, seq_len, n_channels)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_test, seq_len, n_channels)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


def train_random_forest(X_train, y_train, X_test):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test).max(axis=1)

    return preds, probs


def train_mlp(X_train, y_train, X_test):
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=RANDOM_SEED,
        early_stopping=True,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test).max(axis=1)

    return preds, probs


if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.25,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.25)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.norm(out[:, -1, :])
            out = self.dropout(out)
            return self.fc(out)


    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.25,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.25)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.norm(out[:, -1, :])
            out = self.dropout(out)
            return self.fc(out)


    class TransformerModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, max_len):
            super().__init__()

            self.input_proj = nn.Linear(input_size, hidden_size)
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_size))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 2,
                dropout=0.25,
                batch_first=True,
                activation="gelu",
            )

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.norm = nn.LayerNorm(hidden_size)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.input_proj(x)
            x = x + self.pos_embedding[:, :x.size(1), :]
            x = self.encoder(x)
            x = self.norm(x.mean(dim=1))
            return self.fc(x)


def get_device():
    if not TORCH_AVAILABLE:
        return None

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train_torch_model(model, X_train, y_train, X_test):
    device = get_device()
    model = model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    class_counts = np.bincount(y_train, minlength=len(LABEL_NAMES)).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    wait = 0

    for _ in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / max(len(train_loader), 1)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    preds = probabilities.argmax(axis=1)
    confs = probabilities.max(axis=1)

    return preds, confs


def evaluate_models(X_seq, X_stat, y, groups):
    logo = LeaveOneGroupOut()

    model_names = ["RandomForest", "MLP"]

    if TORCH_AVAILABLE:
        model_names += ["LSTM", "GRU", "Transformer"]

    predictions = {name: np.zeros_like(y) for name in model_names}
    confidences = {name: np.zeros(len(y), dtype=float) for name in model_names}
    fold_scores = {name: [] for name in model_names}

    subjects = []

    for fold_id, (train_idx, test_idx) in enumerate(logo.split(X_seq, y, groups)):
        subject = groups[test_idx[0]]
        subjects.append(subject)

        X_train_stat, X_test_stat = scale_stat_data(X_stat[train_idx], X_stat[test_idx])
        X_train_seq, X_test_seq = scale_sequence_data(X_seq[train_idx], X_seq[test_idx])

        y_train = y[train_idx]
        y_test = y[test_idx]

        rf_preds, rf_confs = train_random_forest(X_train_stat, y_train, X_test_stat)
        predictions["RandomForest"][test_idx] = rf_preds
        confidences["RandomForest"][test_idx] = rf_confs
        fold_scores["RandomForest"].append(accuracy_score(y_test, rf_preds))

        mlp_preds, mlp_confs = train_mlp(X_train_stat, y_train, X_test_stat)
        predictions["MLP"][test_idx] = mlp_preds
        confidences["MLP"][test_idx] = mlp_confs
        fold_scores["MLP"].append(accuracy_score(y_test, mlp_preds))

        if TORCH_AVAILABLE:
            lstm = LSTMModel(input_size=3, hidden_size=96, num_classes=len(LABEL_NAMES))
            lstm_preds, lstm_confs = train_torch_model(lstm, X_train_seq, y_train, X_test_seq)
            predictions["LSTM"][test_idx] = lstm_preds
            confidences["LSTM"][test_idx] = lstm_confs
            fold_scores["LSTM"].append(accuracy_score(y_test, lstm_preds))

            gru = GRUModel(input_size=3, hidden_size=96, num_classes=len(LABEL_NAMES))
            gru_preds, gru_confs = train_torch_model(gru, X_train_seq, y_train, X_test_seq)
            predictions["GRU"][test_idx] = gru_preds
            confidences["GRU"][test_idx] = gru_confs
            fold_scores["GRU"].append(accuracy_score(y_test, gru_preds))

            transformer = TransformerModel(
                input_size=3,
                hidden_size=96,
                num_classes=len(LABEL_NAMES),
                max_len=WINDOW_SIZE,
            )
            tr_preds, tr_confs = train_torch_model(transformer, X_train_seq, y_train, X_test_seq)
            predictions["Transformer"][test_idx] = tr_preds
            confidences["Transformer"][test_idx] = tr_confs
            fold_scores["Transformer"].append(accuracy_score(y_test, tr_preds))

        print(f"Fold {fold_id + 1:02d} | left subject: {subject}")

        for name in model_names:
            print(f"  {name:<13}: {fold_scores[name][-1]:.3f}")

    return model_names, predictions, confidences, fold_scores, subjects

def save_fold_scores_csv(fold_scores, subjects):
    rows = []
    for model_name, scores in fold_scores.items():
        for subject, score in zip(subjects, scores):
            rows.append({"model": model_name, "subject": subject,
                        "loso_accuracy": round(float(score), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "fold_scores_long.csv"), index=False)
    return df

def save_model_comparison(fold_scores):
    rows = []

    for model_name, scores in fold_scores.items():
        scores = np.array(scores)

        rows.append({
            "model": model_name,
            "mean_loso_accuracy": round(float(scores.mean()), 4),
            "std_loso_accuracy": round(float(scores.std()), 4),
            "min_loso_accuracy": round(float(scores.min()), 4),
            "max_loso_accuracy": round(float(scores.max()), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    return df


def save_comparison_plot(df_comparison):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        df_comparison["model"],
        df_comparison["mean_loso_accuracy"],
        yerr=df_comparison["std_loso_accuracy"],
        capsize=4,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("LOSO accuracy")
    ax.set_title("Model comparison on RUN - Leave-One-Subject-Out")

    for i, row in df_comparison.iterrows():
        ax.text(
            i,
            row["mean_loso_accuracy"] + 0.02,
            f"{row['mean_loso_accuracy']:.2f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_model_comparison.png"), dpi=150)
    plt.close()


def save_confusion_matrices(y, predictions):
    for model_name, preds in predictions.items():
        fig, ax = plt.subplots(figsize=(8, 7))

        ConfusionMatrixDisplay(
            confusion_matrix(y, preds),
            display_labels=[state.replace("_", "\n") for state in LABEL_NAMES],
        ).plot(ax=ax, cmap="Blues", colorbar=False)

        ax.set_title(f"Confusion Matrix - {model_name} - RUN LOSO")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_{model_name}.png"), dpi=150)
        plt.close()


def build_results_csv(y, groups, meta, predictions, confidences):
    rows = []

    for model_name, preds in predictions.items():
        temporal_scores = temporal_consistency_score(preds, groups)

        for idx in range(len(y)):
            confidence = float(confidences[model_name][idx])
            temporal_score = float(temporal_scores[idx])

            rows.append({
                "model": model_name,
                "window_id": idx,
                "subject": groups[idx],
                "crd": meta[idx]["crd"],
                "true_state": LABEL_NAMES[y[idx]],
                "predicted_state": LABEL_NAMES[preds[idx]],
                "confidence": round(confidence, 4),
                "temporal_consistency": round(temporal_score, 4),
                "hitl_flag": hitl_flag(confidence, temporal_score),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "results_deep_learning_run.csv"), index=False)

    return df


def save_hitl_plot(df_results):
    hitl_counts = (
        df_results
        .groupby(["model", "hitl_flag"])
        .size()
        .unstack(fill_value=0)
    )

    order = ["accept", "check", "review"]

    for col in order:
        if col not in hitl_counts.columns:
            hitl_counts[col] = 0

    hitl_counts = hitl_counts[order]

    fig, ax = plt.subplots(figsize=(9, 5))

    bottom = np.zeros(len(hitl_counts))

    for flag in order:
        values = hitl_counts[flag].values
        ax.bar(hitl_counts.index, values, bottom=bottom, label=flag)
        bottom += values

    ax.set_title("Human-in-the-loop flags by model")
    ax.set_ylabel("Number of windows")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_hitl_flags.png"), dpi=150)
    plt.close()


def save_temporal_consistency_plot(df_results):
    summary = (
        df_results
        .groupby("model")["temporal_consistency"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(summary["model"], summary["temporal_consistency"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean temporal consistency")
    ax.set_title("Temporal consistency by model")

    for i, row in summary.iterrows():
        ax.text(
            i,
            row["temporal_consistency"] + 0.02,
            f"{row['temporal_consistency']:.2f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_temporal_consistency.png"), dpi=150)
    plt.close()


def main():

    X_seq, X_stat, y, groups, meta, used_files, skipped_files = load_run_windows()

    if len(X_seq) == 0:
        raise RuntimeError("No RUN windows found. Check RUN_ROOT path.")

    print("")
    print("Dataset")
    print(f"  RUN path: {RUN_ROOT}")
    print(f"  Files used: {len(used_files)}")
    print(f"  Subjects: {len(set(groups))}")
    print(f"  Windows: {len(y)}")
    print(f"  Sequence input: {X_seq.shape}")
    print(f"  Statistical input: {X_stat.shape}")
    print(f"  Class distribution: {dict(Counter(y))}")

    if skipped_files:
        print(f"  Skipped files: {len(skipped_files)}")

    print("")
    print("Backend")
    print(f"  PyTorch available: {TORCH_AVAILABLE}")

    model_names, predictions, confidences, fold_scores, subjects = evaluate_models(
        X_seq,
        X_stat,
        y,
        groups,
    )

    df_comparison = save_model_comparison(fold_scores)
    df_results = build_results_csv(y, groups, meta, predictions, confidences)

    save_comparison_plot(df_comparison)
    save_confusion_matrices(y, predictions)
    save_hitl_plot(df_results)
    save_temporal_consistency_plot(df_results)

    print("")
    print("Final comparison")
    print(df_comparison.to_string(index=False))

    print("")
    print("Generated files")
    print(f"  {OUTPUT_DIR}/model_comparison.csv")
    print(f"  {OUTPUT_DIR}/results_deep_learning_run.csv")
    print(f"  {OUTPUT_DIR}/1_model_comparison.png")
    print(f"  {OUTPUT_DIR}/2_hitl_flags.png")
    print(f"  {OUTPUT_DIR}/3_temporal_consistency.png")
    print(f"  {OUTPUT_DIR}/confusion_<model>.png")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()