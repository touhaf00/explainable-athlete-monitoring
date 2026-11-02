from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from deep_learning_multisport import (
    LABEL_NAMES,
    load_multisport_windows,
)


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "results" / "experiments" / "recovery"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
CALIBRATION_WINDOWS = 5


BASE_FEATURES = [
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


TEMPORAL_FEATURES = [
    "slope_HR_per_sec",
    "slope_BR_per_sec",
    "slope_RR_per_sec",
    "delta_HR",
    "delta_BR",
    "delta_RR",
    "rmssd_RR",
]


PERSONAL_FEATURES = [
    "mean_HR_delta_cal",
    "mean_BR_delta_cal",
    "mean_RR_delta_cal",
    "mean_HR_ratio_cal",
    "mean_BR_ratio_cal",
    "mean_RR_ratio_cal",
]


def slope_per_second(values, duration_seconds):
    values = np.asarray(values, dtype=float)

    if len(values) < 2 or not np.isfinite(values).all():
        return 0.0

    duration_seconds = max(
        float(duration_seconds),
        1e-6,
    )

    t = np.linspace(
        0.0,
        duration_seconds,
        len(values),
    )

    return float(
        np.polyfit(
            t,
            values,
            1,
        )[0]
    )


def edge_delta(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    k = max(
        1,
        min(
            10,
            len(values) // 5,
        ),
    )

    return float(
        np.mean(values[-k:])
        - np.mean(values[:k])
    )


def build_feature_table(
    x_seq,
    x_stat,
    groups,
    sports,
    meta,
):
    rows = []

    for i, (
        seq,
        stat,
        group,
        sport,
        info,
    ) in enumerate(
        zip(
            x_seq,
            x_stat,
            groups,
            sports,
            meta,
        )
    ):
        hr = seq[:, 0]
        br = seq[:, 1]
        rr = seq[:, 2]

        duration = (
            float(
                info.get(
                    "window_end_sec",
                    0.0,
                )
            )
            - float(
                info.get(
                    "window_start_sec",
                    0.0,
                )
            )
        )

        if duration <= 0:
            sps = max(
                float(
                    info.get(
                        "samples_per_second",
                        1.0,
                    )
                ),
                1e-6,
            )

            duration = (
                len(hr) / sps
            )

        row = {
            "window_id": i,
            "group": str(group),
            "sport": str(sport),
            "window_start_sec": float(
                info.get(
                    "window_start_sec",
                    i,
                )
            ),
        }

        row.update(
            {
                name: float(value)
                for name, value
                in zip(
                    BASE_FEATURES,
                    stat,
                )
            }
        )

        row.update(
            {
                "slope_HR_per_sec": slope_per_second(
                    hr,
                    duration,
                ),
                "slope_BR_per_sec": slope_per_second(
                    br,
                    duration,
                ),
                "slope_RR_per_sec": slope_per_second(
                    rr,
                    duration,
                ),
                "delta_HR": edge_delta(hr),
                "delta_BR": edge_delta(br),
                "delta_RR": edge_delta(rr),
                "rmssd_RR": float(
                    np.sqrt(
                        np.mean(
                            np.diff(rr) ** 2
                        )
                    )
                )
                if len(rr) > 1
                else 0.0,
            }
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    calibration_rows = []

    for group, part in df.groupby(
        "group",
        sort=False,
    ):
        first = (
            part
            .sort_values(
                "window_start_sec"
            )
            .head(
                CALIBRATION_WINDOWS
            )
        )

        calibration_rows.append(
            {
                "group": group,
                "cal_HR": float(
                    first[
                        "mean_HR"
                    ].median()
                ),
                "cal_BR": float(
                    first[
                        "mean_BR"
                    ].median()
                ),
                "cal_RR": float(
                    first[
                        "mean_RR"
                    ].median()
                ),
            }
        )

    calibration = pd.DataFrame(
        calibration_rows
    )

    df = df.merge(
        calibration,
        on="group",
        how="left",
    )

    eps = 1e-6

    df["mean_HR_delta_cal"] = (
        df["mean_HR"]
        - df["cal_HR"]
    )

    df["mean_BR_delta_cal"] = (
        df["mean_BR"]
        - df["cal_BR"]
    )

    df["mean_RR_delta_cal"] = (
        df["mean_RR"]
        - df["cal_RR"]
    )

    df["mean_HR_ratio_cal"] = (
        df["mean_HR"]
        / (
            np.abs(
                df["cal_HR"]
            )
            + eps
        )
    )

    df["mean_BR_ratio_cal"] = np.where(
        np.abs(
            df["cal_BR"]
        ) > eps,
        df["mean_BR"]
        / (
            np.abs(
                df["cal_BR"]
            )
            + eps
        ),
        1.0,
    )

    df["mean_RR_ratio_cal"] = (
        df["mean_RR"]
        / (
            np.abs(
                df["cal_RR"]
            )
            + eps
        )
    )

    return df


def make_models():
    return {
        "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),

        "MLP":
        MLPClassifier(
            hidden_layer_sizes=(
                128,
                64,
            ),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=RANDOM_SEED,
            early_stopping=True,
        ),
    }


def evaluate_feature_set(
    df,
    y,
    groups,
    feature_set_name,
    feature_names,
):
    x = df[
        feature_names
    ].to_numpy(
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    groups = np.asarray(
        groups,
        dtype=str,
    )

    splitter = LeaveOneGroupOut()

    pred_by_model = {
        name: np.full(
            len(y),
            -1,
            dtype=int,
        )
        for name
        in make_models()
    }

    fold_rows = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            x,
            y,
            groups,
        ),
        start=1,
    ):
        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            x[train_idx]
        )

        x_test = scaler.transform(
            x[test_idx]
        )

        for (
            model_name,
            model,
        ) in make_models().items():

            model.fit(
                x_train,
                y[train_idx],
            )

            pred = model.predict(
                x_test
            )

            pred_by_model[
                model_name
            ][test_idx] = pred

            fold_rows.append(
                {
                    "feature_set":
                    feature_set_name,

                    "model":
                    model_name,

                    "fold":
                    fold,

                    "held_out_group":
                    groups[
                        test_idx[0]
                    ],

                    "accuracy":
                    accuracy_score(
                        y[test_idx],
                        pred,
                    ),

                    "balanced_accuracy":
                    balanced_accuracy_score(
                        y[test_idx],
                        pred,
                    ),
                }
            )

    summary_rows = []
    prediction_rows = []

    for (
        model_name,
        pred,
    ) in pred_by_model.items():

        valid = pred >= 0

        yy = y[valid]
        pp = pred[valid]

        recalls = recall_score(
            yy,
            pp,
            labels=[
                0,
                1,
                2,
            ],
            average=None,
            zero_division=0,
        )

        summary_rows.append(
            {
                "feature_set":
                feature_set_name,

                "model":
                model_name,

                "n_features":
                len(
                    feature_names
                ),

                "pooled_accuracy":
                accuracy_score(
                    yy,
                    pp,
                ),

                "pooled_balanced_accuracy":
                balanced_accuracy_score(
                    yy,
                    pp,
                ),

                "macro_f1":
                f1_score(
                    yy,
                    pp,
                    average="macro",
                    zero_division=0,
                ),

                "recall_rest":
                recalls[0],

                "recall_effort":
                recalls[1],

                "recall_recovery":
                recalls[2],
            }
        )

        for idx in np.where(
            valid
        )[0]:
            prediction_rows.append(
                {
                    "feature_set":
                    feature_set_name,

                    "model":
                    model_name,

                    "window_id":
                    int(idx),

                    "group":
                    groups[idx],

                    "true_state":
                    LABEL_NAMES[
                        int(
                            y[idx]
                        )
                    ],

                    "predicted_state":
                    LABEL_NAMES[
                        int(
                            pred[idx]
                        )
                    ],
                }
            )

    return (
        fold_rows,
        summary_rows,
        prediction_rows,
    )


def main():
    (
        x_seq,
        x_stat,
        y,
        groups,
        sports,
        meta,
        _,
        _,
        _,
    ) = load_multisport_windows()

    feature_df = build_feature_table(
        x_seq,
        x_stat,
        groups,
        sports,
        meta,
    )

    feature_df[
        "true_state"
    ] = [
        LABEL_NAMES[
            int(value)
        ]
        for value in y
    ]

    feature_df.to_csv(
        OUTPUT_DIR
        / "recovery_feature_table.csv",
        index=False,
    )

    feature_sets = {
        "base_9":
        BASE_FEATURES,

        "temporal_16":
        BASE_FEATURES
        + TEMPORAL_FEATURES,

        "personalized_22":
        BASE_FEATURES
        + TEMPORAL_FEATURES
        + PERSONAL_FEATURES,
    }

    all_folds = []
    all_summary = []
    all_predictions = []

    for (
        name,
        features,
    ) in feature_sets.items():

        (
            folds,
            summary,
            predictions,
        ) = evaluate_feature_set(
            feature_df,
            y,
            groups,
            name,
            features,
        )

        all_folds.extend(
            folds
        )

        all_summary.extend(
            summary
        )

        all_predictions.extend(
            predictions
        )

    fold_df = pd.DataFrame(
        all_folds
    )

    summary_df = pd.DataFrame(
        all_summary
    )

    prediction_df = pd.DataFrame(
        all_predictions
    )

    fold_df.to_csv(
        OUTPUT_DIR
        / "fold_scores.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "feature_set_summary.csv",
        index=False,
    )

    prediction_df.to_csv(
        OUTPUT_DIR
        / "predictions.csv",
        index=False,
    )

    print(
        summary_df
        .sort_values(
            [
                "model",
                "recall_recovery",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
