import os
import re
import warnings
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPORTDB_ROOT = os.path.join(BASE_DIR, "SportDB2")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_dl_multisport")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
WINDOW_SIZE = 50
STRIDE = 25

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 6

LABEL_NAMES = ["S0_Rest", "S1_Effort", "S2_Recovery"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}

CHANNEL_NAMES = ["HR", "BR", "RR"]

ALLOWED_TRANSITIONS = {
    0: [0, 1],
    1: [1, 2],
    2: [2],
}

SPORTS_CONFIG = {
    "RUN": {
        "encoding": "latin-1",
        "has_br": True,
        "state_map": {
            "resting": "S0_Rest",
            "i phase": "S1_Effort",
            "ii phase": "S1_Effort",
            "iii phase": "S1_Effort",
            "iv phase": "S1_Effort",
            "recovery": "S2_Recovery",
        },
    },
    "CYC": {
        "encoding": "utf-8-sig",
        "has_br": True,
        "state_map": {
            "resting": "S0_Rest",
            "exercise": "S1_Effort",
            "recovery": "S2_Recovery",
        },
    },
    "TRE": {
        "encoding": "latin-1",
        "has_br": True,
        "state_map": {
            "trekking": "S1_Effort",
        },
    },
    "SKA": {
        "encoding": "latin-1",
        "has_br": False,
        "state_map": {
            "resting": "S0_Rest",
            "warm-up": "S1_Effort",
            "warm up": "S1_Effort",
            "laps": "S1_Effort",
            "20 laps": "S1_Effort",
            "recovery": "S2_Recovery",
        },
    },
}

np.random.seed(RANDOM_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)


def hms_to_seconds(value):
    parts = value.strip().split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_trnote(path, encoding, state_map):
    try:
        with open(path, "r", encoding=encoding, errors="replace") as file:
            text = file.read()
    except Exception:
        return {}

    pattern = re.compile(
        r"([A-Za-z0-9\s\-_]+?)\s*:\s*from\s+"
        r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s+to\s+"
        r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",
        re.IGNORECASE,
    )

    segments = {}

    for match in pattern.finditer(text):
        raw_label = match.group(1).strip().lower()
        start = hms_to_seconds(match.group(2))
        end = hms_to_seconds(match.group(3))

        for key, unified_label in state_map.items():
            if key in raw_label:
                if unified_label not in segments:
                    segments[unified_label] = (start, end)
                else:
                    old_start, old_end = segments[unified_label]
                    segments[unified_label] = (
                        min(old_start, start),
                        max(old_end, end),
                    )
                break

    return segments


def get_stat_features(hr, br, rr):
    return [
        np.mean(hr), np.std(hr), np.max(hr),
        np.mean(br), np.std(br), np.max(br),
        np.mean(rr), np.std(rr), np.min(rr),
    ]


def load_multisport_windows():
    x_seq = []
    x_stat = []
    y = []
    groups = []
    sports = []
    meta = []

    used_files = []
    skipped_files = []
    windows_per_sport = defaultdict(int)

    for sport, config in SPORTS_CONFIG.items():
        sport_path = os.path.join(SPORTDB_ROOT, sport)

        if not os.path.isdir(sport_path):
            print(f"[{sport}] missing folder: {sport_path}")
            continue

        for dirpath, _, filenames in os.walk(sport_path):
            if "Data.mat" not in filenames or "TrNote.txt" not in filenames:
                continue

            data_path = os.path.join(dirpath, "Data.mat")
            note_path = os.path.join(dirpath, "TrNote.txt")
            relative_subject = os.path.relpath(dirpath, sport_path)
            group_id = f"{sport}_{relative_subject.replace(os.sep, '_')}"

            try:
                mat_data = loadmat(data_path)
                data = mat_data["Data"]
                fields = data.dtype.names

                if "HR" not in fields or "RR" not in fields:
                    skipped_files.append((sport, relative_subject, "missing HR/RR"))
                    continue

                if config["has_br"] and "BR" not in fields:
                    skipped_files.append((sport, relative_subject, "missing BR"))
                    continue

                hr = data[0, 0]["HR"].flatten().astype(float)
                rr = data[0, 0]["RR"].flatten().astype(float)

                if config["has_br"]:
                    br = data[0, 0]["BR"].flatten().astype(float)
                else:
                    br = np.zeros_like(hr)

                segments = parse_trnote(
                    note_path,
                    config["encoding"],
                    config["state_map"],
                )

                if not segments:
                    skipped_files.append((sport, relative_subject, "no parsed segment"))
                    continue

                total_duration = max(end for _, end in segments.values())

                if total_duration <= 0:
                    skipped_files.append((sport, relative_subject, "invalid duration"))
                    continue

                samples_per_second = len(hr) / total_duration

                for state_name, (start_sec, end_sec) in segments.items():
                    start = int(start_sec * samples_per_second)
                    end = int(end_sec * samples_per_second)
                    label = LABEL_TO_ID[state_name]

                    for i in range(start, end - WINDOW_SIZE, STRIDE):
                        j = i + WINDOW_SIZE

                        hr_window = hr[i:j]
                        br_window = br[i:j]
                        rr_window = rr[i:j]

                        if len(hr_window) != WINDOW_SIZE:
                            continue

                        valid = (
                            np.isfinite(hr_window).all()
                            and np.isfinite(br_window).all()
                            and np.isfinite(rr_window).all()
                        )

                        if not valid:
                            continue

                        x_seq.append(
                            np.stack([hr_window, br_window, rr_window], axis=1)
                        )
                        x_stat.append(get_stat_features(hr_window, br_window, rr_window))
                        y.append(label)
                        groups.append(group_id)
                        sports.append(sport)

                        meta.append({
                            "sport": sport,
                            "subject": relative_subject,
                            "state": state_name,
                        })

                        windows_per_sport[sport] += 1

                used_files.append((sport, relative_subject))

            except Exception as error:
                skipped_files.append((sport, relative_subject, str(error)))

    return (
        np.array(x_seq, dtype=np.float32),
        np.array(x_stat, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(groups),
        np.array(sports),
        meta,
        used_files,
        skipped_files,
        windows_per_sport,
    )


def scale_stat_data(x_train, x_test):
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def scale_sequence_data(x_train, x_test):
    scaler = StandardScaler()

    n_train, seq_len, n_channels = x_train.shape
    n_test = x_test.shape[0]

    train_flat = x_train.reshape(-1, n_channels)
    test_flat = x_test.reshape(-1, n_channels)

    train_scaled = scaler.fit_transform(train_flat)
    test_scaled = scaler.transform(test_flat)

    return (
        train_scaled.reshape(n_train, seq_len, n_channels).astype(np.float32),
        test_scaled.reshape(n_test, seq_len, n_channels).astype(np.float32),
    )


def train_random_forest(x_train, y_train, x_test):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    confidence = model.predict_proba(x_test).max(axis=1)

    return predictions, confidence


def train_mlp(x_train, y_train, x_test):
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=RANDOM_SEED,
        early_stopping=True,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    confidence = model.predict_proba(x_test).max(axis=1)

    return predictions, confidence


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
            output, _ = self.lstm(x)
            output = output[:, -1, :]
            output = self.norm(output)
            output = self.dropout(output)
            return self.fc(output)


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
            output, _ = self.gru(x)
            output = output[:, -1, :]
            output = self.norm(output)
            output = self.dropout(output)
            return self.fc(output)


    class TransformerModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, max_len):
            super().__init__()

            self.input_projection = nn.Linear(input_size, hidden_size)
            self.position_embedding = nn.Parameter(
                torch.randn(1, max_len, hidden_size)
            )

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
            x = self.input_projection(x)
            x = x + self.position_embedding[:, :x.size(1), :]
            x = self.encoder(x)
            x = x.mean(dim=1)
            x = self.norm(x)
            return self.fc(x)


def get_device():
    if not TORCH_AVAILABLE:
        return None

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train_torch_model(model, x_train, y_train, x_test, num_classes):
    device = get_device()
    model = model.to(device)

    train_data = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

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
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    return predictions, confidence


def temporal_consistency_score(predictions, groups):
    scores = np.ones(len(predictions), dtype=float)

    for group in sorted(set(groups)):
        indexes = np.where(groups == group)[0]
        group_predictions = predictions[indexes]

        for i in range(1, len(group_predictions)):
            previous_state = int(group_predictions[i - 1])
            current_state = int(group_predictions[i])

            if current_state in ALLOWED_TRANSITIONS.get(previous_state, []):
                scores[indexes[i]] = 1.0
            else:
                distance = abs(current_state - previous_state)
                scores[indexes[i]] = 0.5 if distance <= 1 else 0.0

    return scores


def hitl_flag(confidence, temporal_score):
    if confidence < 0.45 or temporal_score < 0.5:
        return "review"

    if confidence < 0.60 or temporal_score < 1.0:
        return "check"

    return "accept"


def evaluate_models(x_seq, x_stat, y, groups, sports):
    logo = LeaveOneGroupOut()

    model_names = ["RandomForest", "MLP"]

    if TORCH_AVAILABLE:
        model_names += ["LSTM", "GRU", "Transformer"]

    predictions = {
        name: np.zeros_like(y)
        for name in model_names
    }

    confidences = {
        name: np.zeros(len(y), dtype=float)
        for name in model_names
    }

    fold_scores = {
        name: []
        for name in model_names
    }

    fold_subjects = []
    fold_sports = []

    for fold_id, (train_idx, test_idx) in enumerate(logo.split(x_seq, y, groups)):
        subject = groups[test_idx[0]]
        sport = sports[test_idx[0]]

        fold_subjects.append(subject)
        fold_sports.append(sport)

        if len(set(y[train_idx])) < 2:
            print(f"Fold {fold_id + 1:02d} | {subject} skipped")
            for name in model_names:
                fold_scores[name].append(np.nan)
            continue

        x_train_stat, x_test_stat = scale_stat_data(
            x_stat[train_idx],
            x_stat[test_idx],
        )

        x_train_seq, x_test_seq = scale_sequence_data(
            x_seq[train_idx],
            x_seq[test_idx],
        )

        y_train = y[train_idx]
        y_test = y[test_idx]

        rf_pred, rf_conf = train_random_forest(x_train_stat, y_train, x_test_stat)
        predictions["RandomForest"][test_idx] = rf_pred
        confidences["RandomForest"][test_idx] = rf_conf
        fold_scores["RandomForest"].append(accuracy_score(y_test, rf_pred))

        mlp_pred, mlp_conf = train_mlp(x_train_stat, y_train, x_test_stat)
        predictions["MLP"][test_idx] = mlp_pred
        confidences["MLP"][test_idx] = mlp_conf
        fold_scores["MLP"].append(accuracy_score(y_test, mlp_pred))

        if TORCH_AVAILABLE:
            lstm = LSTMModel(3, 96, len(LABEL_NAMES))
            lstm_pred, lstm_conf = train_torch_model(
                lstm,
                x_train_seq,
                y_train,
                x_test_seq,
                len(LABEL_NAMES),
            )
            predictions["LSTM"][test_idx] = lstm_pred
            confidences["LSTM"][test_idx] = lstm_conf
            fold_scores["LSTM"].append(accuracy_score(y_test, lstm_pred))

            gru = GRUModel(3, 96, len(LABEL_NAMES))
            gru_pred, gru_conf = train_torch_model(
                gru,
                x_train_seq,
                y_train,
                x_test_seq,
                len(LABEL_NAMES),
            )
            predictions["GRU"][test_idx] = gru_pred
            confidences["GRU"][test_idx] = gru_conf
            fold_scores["GRU"].append(accuracy_score(y_test, gru_pred))

            transformer = TransformerModel(3, 96, len(LABEL_NAMES), WINDOW_SIZE)
            transformer_pred, transformer_conf = train_torch_model(
                transformer,
                x_train_seq,
                y_train,
                x_test_seq,
                len(LABEL_NAMES),
            )
            predictions["Transformer"][test_idx] = transformer_pred
            confidences["Transformer"][test_idx] = transformer_conf
            fold_scores["Transformer"].append(
                accuracy_score(y_test, transformer_pred)
            )

        print(f"Fold {fold_id + 1:02d} | {subject:<22} ({sport})")

        for name in model_names:
            last_score = fold_scores[name][-1]

            if np.isnan(last_score):
                print(f"  {name:<13}: skipped")
            else:
                print(f"  {name:<13}: {last_score:.3f}")

    return model_names, predictions, confidences, fold_scores, fold_subjects, fold_sports


def save_model_comparison(fold_scores):
    rows = []

    for model_name, scores in fold_scores.items():
        valid_scores = np.array([score for score in scores if not np.isnan(score)])

        rows.append({
            "model": model_name,
            "mean_loso_accuracy": round(float(valid_scores.mean()), 4),
            "std_loso_accuracy": round(float(valid_scores.std()), 4),
            "min_loso_accuracy": round(float(valid_scores.min()), 4),
            "max_loso_accuracy": round(float(valid_scores.max()), 4),
            "n_folds": len(valid_scores),
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison_multisport.csv"),
        index=False,
    )

    return df


def save_fold_scores(fold_scores, fold_subjects, fold_sports):
    rows = []

    for model_name, scores in fold_scores.items():
        for subject, sport, score in zip(fold_subjects, fold_sports, scores):
            rows.append({
                "model": model_name,
                "subject": subject,
                "sport": sport,
                "loso_accuracy": round(float(score), 4)
                if not np.isnan(score)
                else np.nan,
            })

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUTPUT_DIR, "fold_scores_multisport.csv"),
        index=False,
    )

    return df


def save_comparison_plot(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        df["model"],
        df["mean_loso_accuracy"],
        yerr=df["std_loso_accuracy"],
        capsize=4,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("LOSO accuracy")
    ax.set_title("Multi-sport model comparison")

    for i, row in df.iterrows():
        ax.text(
            i,
            row["mean_loso_accuracy"] + 0.02,
            f"{row['mean_loso_accuracy']:.2f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "1_model_comparison_multisport.png"),
        dpi=150,
    )
    plt.close()


def save_confusion_matrices(y, predictions):
    for model_name, pred in predictions.items():
        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay(
            confusion_matrix(y, pred),
            display_labels=[label.replace("_", "\n") for label in LABEL_NAMES],
        ).plot(ax=ax, cmap="Blues", colorbar=False)

        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"confusion_{model_name}_multisport.png"),
            dpi=150,
        )
        plt.close()


def build_results_table(y, groups, sports, meta, predictions, confidences):
    rows = []

    for model_name, pred in predictions.items():
        temp_scores = temporal_consistency_score(pred, groups)

        for i in range(len(y)):
            confidence = float(confidences[model_name][i])
            temp_score = float(temp_scores[i])

            rows.append({
                "model": model_name,
                "window_id": i,
                "sport": sports[i],
                "subject": groups[i],
                "true_state": LABEL_NAMES[y[i]],
                "predicted_state": LABEL_NAMES[pred[i]],
                "confidence": round(confidence, 4),
                "temporal_consistency": round(temp_score, 4),
                "hitl_flag": hitl_flag(confidence, temp_score),
            })

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUTPUT_DIR, "results_multisport.csv"),
        index=False,
    )

    return df


def save_accuracy_by_sport(df):
    df = df.copy()
    df["correct"] = (
        df["true_state"] == df["predicted_state"]
    ).astype(int)

    summary = (
        df.groupby(["sport", "model"])["correct"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    summary.plot(kind="bar", ax=ax, width=0.8)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by sport and model")
    ax.legend(title="Model", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "4_per_sport_accuracy.png"),
        dpi=150,
    )
    plt.close()


def save_hitl_plot(df):
    counts = (
        df.groupby(["model", "hitl_flag"])
        .size()
        .unstack(fill_value=0)
    )

    order = ["accept", "check", "review"]

    for col in order:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts[order]

    fig, ax = plt.subplots(figsize=(9, 5))

    bottom = np.zeros(len(counts))

    for flag in order:
        values = counts[flag].values
        ax.bar(counts.index, values, bottom=bottom, label=flag)
        bottom += values

    ax.set_title("Human-in-the-loop flags")
    ax.set_ylabel("Number of windows")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "2_hitl_flags_multisport.png"),
        dpi=150,
    )
    plt.close()


def save_temporal_consistency_plot(df):
    summary = (
        df.groupby("model")["temporal_consistency"]
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
    plt.savefig(
        os.path.join(OUTPUT_DIR, "3_temporal_consistency_multisport.png"),
        dpi=150,
    )
    plt.close()


def main():
    print("=" * 70)
    print("Multi-sport temporal learning")
    print("Sports: RUN, CYC, TRE, SKA")
    print("=" * 70)

    (
        x_seq,
        x_stat,
        y,
        groups,
        sports,
        meta,
        used_files,
        skipped_files,
        windows_per_sport,
    ) = load_multisport_windows()

    if len(x_seq) == 0:
        raise RuntimeError("No windows were loaded. Check SportDB2 path.")

    print("")
    print("Dataset")
    print(f"  Files used: {len(used_files)}")
    print(f"  Files skipped: {len(skipped_files)}")
    print(f"  Groups: {len(set(groups))}")
    print(f"  Windows: {len(y)}")
    print(f"  Class distribution: {dict(Counter(y))}")

    print("")
    print("Windows by sport")
    for sport, count in windows_per_sport.items():
        print(f"  {sport:<6}: {count}")

    if skipped_files:
        print("")
        print("First skipped files")
        for item in skipped_files[:10]:
            print(f"  {item}")

    print("")
    print(f"PyTorch available: {TORCH_AVAILABLE}")

    (
        model_names,
        predictions,
        confidences,
        fold_scores,
        fold_subjects,
        fold_sports,
    ) = evaluate_models(x_seq, x_stat, y, groups, sports)

    comparison_df = save_model_comparison(fold_scores)
    save_fold_scores(fold_scores, fold_subjects, fold_sports)

    results_df = build_results_table(
        y,
        groups,
        sports,
        meta,
        predictions,
        confidences,
    )

    save_comparison_plot(comparison_df)
    save_confusion_matrices(y, predictions)
    save_accuracy_by_sport(results_df)
    save_hitl_plot(results_df)
    save_temporal_consistency_plot(results_df)

    print("")
    print("Final comparison")
    print(comparison_df.to_string(index=False))

    print("")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("  model_comparison_multisport.csv")
    print("  fold_scores_multisport.csv")
    print("  results_multisport.csv")
    print("  1_model_comparison_multisport.png")
    print("  2_hitl_flags_multisport.png")
    print("  3_temporal_consistency_multisport.png")
    print("  4_per_sport_accuracy.png")
    print("  confusion_<model>_multisport.png")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()