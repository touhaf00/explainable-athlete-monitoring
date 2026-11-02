import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_ablation")
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

EVAL_SPORTS = ["RUN", "CYC", "TRE"]
EXTRA_TRAIN_SPORT = "SKA"

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
    h, m, s = value.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


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

        for key, state in state_map.items():
            if key in raw_label:
                if state not in segments:
                    segments[state] = (start, end)
                else:
                    old_start, old_end = segments[state]
                    segments[state] = (min(old_start, start), max(old_end, end))
                break

    return segments


def stat_features(hr, br, rr):
    return [
        np.mean(hr), np.std(hr), np.max(hr),
        np.mean(br), np.std(br), np.max(br),
        np.mean(rr), np.std(rr), np.min(rr),
    ]


def load_sport_windows(sport):
    config = SPORTS_CONFIG[sport]
    sport_path = os.path.join(SPORTDB_ROOT, sport)

    x_seq = []
    x_stat = []
    y = []
    groups = []

    if not os.path.isdir(sport_path):
        return (
            np.empty((0, WINDOW_SIZE, 3), dtype=np.float32),
            np.empty((0, 9), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=object),
        )

    for dirpath, _, filenames in os.walk(sport_path):
        if "Data.mat" not in filenames or "TrNote.txt" not in filenames:
            continue

        data_path = os.path.join(dirpath, "Data.mat")
        note_path = os.path.join(dirpath, "TrNote.txt")
        rel_path = os.path.relpath(dirpath, sport_path)
        group_id = f"{sport}_{rel_path.replace(os.sep, '_')}"

        try:
            data = loadmat(data_path)["Data"]
            fields = data.dtype.names

            if "HR" not in fields or "RR" not in fields:
                continue

            if config["has_br"] and "BR" not in fields:
                continue

            hr = data[0, 0]["HR"].flatten().astype(float)
            rr = data[0, 0]["RR"].flatten().astype(float)

            if config["has_br"]:
                br = data[0, 0]["BR"].flatten().astype(float)
            else:
                br = np.zeros_like(hr)

            segments = parse_trnote(note_path, config["encoding"], config["state_map"])

            if not segments:
                continue

            total_duration = max(end for _, end in segments.values())

            if total_duration <= 0:
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

                    if not (
                        np.isfinite(hr_window).all()
                        and np.isfinite(br_window).all()
                        and np.isfinite(rr_window).all()
                    ):
                        continue

                    x_seq.append(np.stack([hr_window, br_window, rr_window], axis=1))
                    x_stat.append(stat_features(hr_window, br_window, rr_window))
                    y.append(label)
                    groups.append(group_id)

        except Exception:
            continue

    return (
        np.array(x_seq, dtype=np.float32),
        np.array(x_stat, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(groups),
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
    return model.predict(x_test)


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
    return model.predict(x_test)


if TORCH_AVAILABLE:

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
            out = out[:, -1, :]
            out = self.norm(out)
            out = self.dropout(out)
            return self.fc(out)


    class TransformerModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, max_len):
            super().__init__()
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.position_embedding = nn.Parameter(torch.randn(1, max_len, hidden_size))

            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 2,
                dropout=0.25,
                batch_first=True,
                activation="gelu",
            )

            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
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

    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
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

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / max(len(loader), 1)

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

    return probabilities.argmax(axis=1)


def recovery_recall(y_true, y_pred):
    recovery_id = LABEL_TO_ID["S2_Recovery"]
    mask = y_true == recovery_id

    if mask.sum() == 0:
        return np.nan

    return (y_pred[mask] == y_true[mask]).mean()


def run_models(x_train_seq, x_train_stat, y_train, x_test_seq, x_test_stat):
    x_train_stat, x_test_stat = scale_stat_data(x_train_stat, x_test_stat)
    x_train_seq, x_test_seq = scale_sequence_data(x_train_seq, x_test_seq)

    predictions = {
        "RandomForest": train_random_forest(x_train_stat, y_train, x_test_stat),
        "MLP": train_mlp(x_train_stat, y_train, x_test_stat),
    }

    if TORCH_AVAILABLE:
        predictions["GRU"] = train_torch_model(
            GRUModel(3, 96, len(LABEL_NAMES)),
            x_train_seq,
            y_train,
            x_test_seq,
            len(LABEL_NAMES),
        )

        predictions["Transformer"] = train_torch_model(
            TransformerModel(3, 96, len(LABEL_NAMES), WINDOW_SIZE),
            x_train_seq,
            y_train,
            x_test_seq,
            len(LABEL_NAMES),
        )

    return predictions


def main():
    print("=" * 70)
    print("Ablation study: adding SKA to the training set")
    print("=" * 70)

    eval_seq = []
    eval_stat = []
    eval_y = []
    eval_groups = []

    for sport in EVAL_SPORTS:
        x_seq, x_stat, y, groups = load_sport_windows(sport)

        print(f"{sport}: {len(y)} windows, {len(set(groups))} groups")

        eval_seq.append(x_seq)
        eval_stat.append(x_stat)
        eval_y.append(y)
        eval_groups.append(groups)

    eval_seq = np.concatenate(eval_seq)
    eval_stat = np.concatenate(eval_stat)
    eval_y = np.concatenate(eval_y)
    eval_groups = np.concatenate(eval_groups)

    ska_seq, ska_stat, ska_y, _ = load_sport_windows(EXTRA_TRAIN_SPORT)
    print(f"{EXTRA_TRAIN_SPORT}: {len(ska_y)} windows used only for training")

    rows = []
    subjects = sorted(set(eval_groups))

    for fold_id, subject in enumerate(subjects, start=1):
        test_mask = eval_groups == subject
        train_mask = ~test_mask

        y_test = eval_y[test_mask]

        base_train_seq = eval_seq[train_mask]
        base_train_stat = eval_stat[train_mask]
        base_train_y = eval_y[train_mask]

        conditions = {
            "no_ska": (
                base_train_seq,
                base_train_stat,
                base_train_y,
            ),
            "with_ska": (
                np.concatenate([base_train_seq, ska_seq]),
                np.concatenate([base_train_stat, ska_stat]),
                np.concatenate([base_train_y, ska_y]),
            ),
        }

        for condition, (x_train_seq, x_train_stat, y_train) in conditions.items():
            if len(set(y_train)) < 2:
                continue

            preds_by_model = run_models(
                x_train_seq,
                x_train_stat,
                y_train,
                eval_seq[test_mask],
                eval_stat[test_mask],
            )

            for model_name, preds in preds_by_model.items():
                rows.append({
                    "subject": subject,
                    "condition": condition,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, preds),
                    "recall_S2_Recovery": recovery_recall(y_test, preds),
                    "n_test": int(test_mask.sum()),
                })

        print(f"Fold {fold_id:02d}/{len(subjects)} | {subject} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)

    summary_rows = []

    for model_name in sorted(df["model"].unique()):
        sub = df[df["model"] == model_name]

        acc = sub.pivot(index="subject", columns="condition", values="accuracy")
        rec = sub.pivot(index="subject", columns="condition", values="recall_S2_Recovery")

        acc_diff = acc["with_ska"] - acc["no_ska"]
        rec = rec.dropna()
        rec_diff = rec["with_ska"] - rec["no_ska"] if len(rec) else pd.Series(dtype=float)

        try:
            _, acc_p = wilcoxon(acc["no_ska"], acc["with_ska"])
        except Exception:
            acc_p = np.nan

        try:
            _, rec_p = wilcoxon(rec["no_ska"], rec["with_ska"]) if len(rec) >= 2 else (np.nan, np.nan)
        except Exception:
            rec_p = np.nan

        summary_rows.append({
            "model": model_name,
            "mean_accuracy_no_ska": acc["no_ska"].mean(),
            "mean_accuracy_with_ska": acc["with_ska"].mean(),
            "accuracy_difference": acc_diff.mean(),
            "accuracy_p_value": acc_p,
            "mean_recall_S2_no_ska": rec["no_ska"].mean() if len(rec) else np.nan,
            "mean_recall_S2_with_ska": rec["with_ska"].mean() if len(rec) else np.nan,
            "recall_S2_difference": rec_diff.mean() if len(rec_diff) else np.nan,
            "recall_S2_p_value": rec_p,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False)

    print("")
    print("Ablation summary")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(summary))
    width = 0.35

    axes[0].bar(x - width / 2, summary["mean_accuracy_no_ska"], width, label="without SKA")
    axes[0].bar(x + width / 2, summary["mean_accuracy_with_ska"], width, label="with SKA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["model"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy on RUN/CYC/TRE")
    axes[0].legend()

    axes[1].bar(x - width / 2, summary["mean_recall_S2_no_ska"], width, label="without SKA")
    axes[1].bar(x + width / 2, summary["mean_recall_S2_with_ska"], width, label="with SKA")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["model"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recovery recall on RUN/CYC/TRE")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_ska_effect.png"), dpi=150)
    plt.close()

    print("")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("  ablation_results.csv")
    print("  ablation_summary.csv")
    print("  ablation_ska_effect.png")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()