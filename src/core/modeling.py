from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from src.core.dataset import (
    load_dataset_v2,
)


CALIBRATION_WINDOWS = 10
RANDOM_SEED = 42


BASE_HR_BR = [
    "mean_HR",
    "std_HR",
    "max_HR",
    "mean_BR",
    "std_BR",
    "max_BR",
]


TEMPORAL_HR_BR = [
    "slope_HR_per_sec",
    "slope_BR_per_sec",
    "delta_HR",
    "delta_BR",
]


PERSONAL_HR_BR = [
    "mean_HR_delta_cal",
    "mean_BR_delta_cal",
    "mean_HR_ratio_cal",
    "mean_BR_ratio_cal",
]


BASE_RR = [
    "mean_RR",
    "std_RR",
    "min_RR",
]


TEMPORAL_RR = [
    "slope_RR_per_sec",
    "delta_RR",
    "rmssd_RR",
]


PERSONAL_RR = [
    "mean_RR_delta_cal",
    "mean_RR_ratio_cal",
]


def slope(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 2:
        return 0.0

    x = np.arange(
        len(values),
        dtype=float,
    )

    return float(
        np.polyfit(
            x,
            values,
            1,
        )[0]
    )


def edge_delta(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    size = max(
        1,
        min(
            10,
            len(values) // 5,
        ),
    )

    return float(
        np.mean(
            values[-size:]
        )
        - np.mean(
            values[:size]
        )
    )


def build_raw_table(
    mode,
):
    (
        x_seq,
        x_stat,
        y,
        groups,
        sports,
        metadata,
        used,
        skipped,
        ignored,
        rr_quality,
    ) = load_dataset_v2(
        mode
    )

    if mode == "hr_br":
        base_names = BASE_HR_BR
    else:
        base_names = (
            BASE_HR_BR
            + BASE_RR
        )

    rows = []

    for (
        sequence,
        stats,
        label,
        group,
        sport,
        item,
    ) in zip(
        x_seq,
        x_stat,
        y,
        groups,
        sports,
        metadata,
    ):
        hr = sequence[:, 0]
        br = sequence[:, 1]

        row = {
            "recording":
            item["recording"],

            "subject":
            str(group),

            "sport":
            str(sport),

            "state":
            int(label),

            "window_start_sample":
            int(
                item[
                    "window_start_sample"
                ]
            ),

            "window_end_sample":
            int(
                item[
                    "window_end_sample"
                ]
            ),
        }

        row.update(
            {
                name:
                float(value)
                for name, value
                in zip(
                    base_names,
                    stats,
                )
            }
        )

        row[
            "slope_HR_per_sec"
        ] = slope(hr)

        row[
            "slope_BR_per_sec"
        ] = slope(br)

        row[
            "delta_HR"
        ] = edge_delta(hr)

        row[
            "delta_BR"
        ] = edge_delta(br)

        if mode == "hr_br_rr":
            rr = sequence[:, 2]

            row[
                "slope_RR_per_sec"
            ] = slope(rr)

            row[
                "delta_RR"
            ] = edge_delta(rr)

            row[
                "rmssd_RR"
            ] = float(
                np.sqrt(
                    np.mean(
                        np.diff(rr) ** 2
                    )
                )
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def add_calibration(
    df,
    mode,
):
    df = df.copy()

    df[
        "is_calibration"
    ] = False

    calibration_rows = []

    for (
        recording,
        part,
    ) in df.groupby(
        "recording",
        sort=False,
    ):
        resting = (
            part[
                part["state"] == 0
            ]
            .sort_values(
                "window_start_sample"
            )
        )

        if (
            len(resting)
            < CALIBRATION_WINDOWS
        ):
            raise ValueError(
                f"{recording}: "
                f"only {len(resting)} "
                f"rest windows"
            )

        calibration = resting.head(
            CALIBRATION_WINDOWS
        )

        df.loc[
            calibration.index,
            "is_calibration",
        ] = True

        row = {
            "recording":
            recording,

            "cal_HR":
            float(
                calibration[
                    "mean_HR"
                ].median()
            ),

            "cal_BR":
            float(
                calibration[
                    "mean_BR"
                ].median()
            ),
        }

        if mode == "hr_br_rr":
            row[
                "cal_RR"
            ] = float(
                calibration[
                    "mean_RR"
                ].median()
            )

        calibration_rows.append(
            row
        )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    df = df.merge(
        calibration_df,
        on="recording",
        how="left",
    )

    eps = 1e-6

    df[
        "mean_HR_delta_cal"
    ] = (
        df["mean_HR"]
        - df["cal_HR"]
    )

    df[
        "mean_BR_delta_cal"
    ] = (
        df["mean_BR"]
        - df["cal_BR"]
    )

    df[
        "mean_HR_ratio_cal"
    ] = (
        df["mean_HR"]
        / (
            np.abs(
                df["cal_HR"]
            )
            + eps
        )
    )

    df[
        "mean_BR_ratio_cal"
    ] = (
        df["mean_BR"]
        / (
            np.abs(
                df["cal_BR"]
            )
            + eps
        )
    )

    if mode == "hr_br_rr":
        df[
            "mean_RR_delta_cal"
        ] = (
            df["mean_RR"]
            - df["cal_RR"]
        )

        df[
            "mean_RR_ratio_cal"
        ] = (
            df["mean_RR"]
            / (
                np.abs(
                    df["cal_RR"]
                )
                + eps
            )
        )

    return df

def select_evaluation_windows(
    df,
):
    calibration_end = (
        df[
            df[
                "is_calibration"
            ]
        ]
        .groupby(
            "recording"
        )[
            "window_end_sample"
        ]
        .max()
        .rename(
            "calibration_end_sample"
        )
    )

    evaluation = df.join(
        calibration_end,
        on="recording",
    )

    evaluation = (
        evaluation[
            (
                ~evaluation[
                    "is_calibration"
                ]
            )
            & (
                evaluation[
                    "window_start_sample"
                ]
                >= evaluation[
                    "calibration_end_sample"
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return evaluation

def feature_sets(
    mode,
):
    if mode == "hr_br":
        return {
            "base_6":
            BASE_HR_BR,

            "temporal_10":
            BASE_HR_BR
            + TEMPORAL_HR_BR,

            "personalized_14":
            BASE_HR_BR
            + TEMPORAL_HR_BR
            + PERSONAL_HR_BR,
        }

    return {
        "base_9":
        BASE_HR_BR
        + BASE_RR,

        "temporal_16":
        BASE_HR_BR
        + BASE_RR
        + TEMPORAL_HR_BR
        + TEMPORAL_RR,

        "personalized_22":
        BASE_HR_BR
        + BASE_RR
        + TEMPORAL_HR_BR
        + TEMPORAL_RR
        + PERSONAL_HR_BR
        + PERSONAL_RR,
    }


def random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def evaluate(
    df,
    experiment,
    mode,
):
    evaluation = select_evaluation_windows(
        df
    )

    y = evaluation[
        "state"
    ].to_numpy(
        dtype=int
    )

    groups = evaluation[
        "subject"
    ].to_numpy(
        dtype=str
    )

    sports = evaluation[
        "sport"
    ].to_numpy(
        dtype=str
    )

    splitter = LeaveOneGroupOut()

    results = []

    predictions = {}

    for (
        feature_name,
        features,
    ) in feature_sets(
        mode
    ).items():
        x = evaluation[
            features
        ].to_numpy(
            dtype=float
        )

        pred = np.full(
            len(y),
            -1,
            dtype=int,
        )

        fold_accuracies = []

        start_time = perf_counter()

        for (
            train_idx,
            test_idx,
        ) in splitter.split(
            x,
            y,
            groups,
        ):
            model = random_forest()

            model.fit(
                x[
                    train_idx
                ],
                y[
                    train_idx
                ],
            )

            fold_pred = model.predict(
                x[
                    test_idx
                ]
            )

            pred[
                test_idx
            ] = fold_pred

            fold_accuracies.append(
                accuracy_score(
                    y[
                        test_idx
                    ],
                    fold_pred,
                )
            )

        elapsed = (
            perf_counter()
            - start_time
        )

        recalls = recall_score(
            y,
            pred,
            labels=[
                0,
                1,
                2,
            ],
            average=None,
            zero_division=0,
        )

        results.append(
            {
                "experiment":
                experiment,

                "mode":
                mode,

                "feature_set":
                feature_name,

                "n_features":
                len(features),

                "n_windows":
                len(y),

                "n_subjects":
                len(
                    set(
                        groups
                    )
                ),

                "accuracy":
                accuracy_score(
                    y,
                    pred,
                ),

                "balanced_accuracy":
                balanced_accuracy_score(
                    y,
                    pred,
                ),

                "macro_f1":
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0,
                ),

                "recall_rest":
                recalls[0],

                "recall_effort":
                recalls[1],

                "recall_recovery":
                recalls[2],

                "mean_fold_accuracy":
                np.mean(
                    fold_accuracies
                ),

                "std_fold_accuracy":
                np.std(
                    fold_accuracies
                ),

                "seconds":
                elapsed,
            }
        )

        predictions[
            feature_name
        ] = {
            "y":
            y.copy(),

            "pred":
            pred.copy(),

            "sports":
            sports.copy(),

            "groups":
            groups.copy(),
        }

    return (
        pd.DataFrame(
            results
        ),
        predictions,
    )


def exact_keys(
    df,
):
    return set(
        zip(
            df[
                "recording"
            ],
            df[
                "window_start_sample"
            ],
            df[
                "state"
            ],
        )
    )


def restrict_to_keys(
    df,
    keys,
):
    mask = [
        (
            recording,
            start,
            state,
        )
        in keys
        for (
            recording,
            start,
            state,
        )
        in zip(
            df[
                "recording"
            ],
            df[
                "window_start_sample"
            ],
            df[
                "state"
            ],
        )
    ]

    return (
        df[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def print_results(
    title,
    frame,
):
    print("")
    print("=" * 110)
    print(title)
    print("=" * 110)

    columns = [
        "experiment",
        "feature_set",
        "n_features",
        "n_windows",
        "n_subjects",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_rest",
        "recall_effort",
        "recall_recovery",
        "mean_fold_accuracy",
        "std_fold_accuracy",
        "seconds",
    ]

    display = frame[
        columns
    ].copy()

    numeric = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_rest",
        "recall_effort",
        "recall_recovery",
        "mean_fold_accuracy",
        "std_fold_accuracy",
        "seconds",
    ]

    display[
        numeric
    ] = display[
        numeric
    ].round(
        4
    )

    print(
        display.to_string(
            index=False
        )
    )


def main():
    print(
        "Loading HR+BR dataset..."
    )

    hr_br_raw = build_raw_table(
        "hr_br"
    )

    hr_br = add_calibration(
        hr_br_raw,
        "hr_br",
    )

    calibration_count = int(
        hr_br[
            "is_calibration"
        ].sum()
    )

    print(
        f"Primary recordings="
        f"{hr_br['recording'].nunique()}"
    )

    print(
        f"Primary subjects="
        f"{hr_br['subject'].nunique()}"
    )

    print(
        f"Calibration windows="
        f"{calibration_count}"
    )

    primary_evaluation = select_evaluation_windows(
        hr_br
    )

    print(
        f"Evaluation windows="
        f"{len(primary_evaluation)}"
    )

    primary_results, _ = evaluate(
        hr_br,
        "primary_all",
        "hr_br",
    )

    print_results(
        "PRIMARY CORPUS — HR + BR",
        primary_results,
    )

    print("")
    print(
        "Loading high-quality RR dataset..."
    )

    rr_raw = build_raw_table(
        "hr_br_rr"
    )

    rr_keys = exact_keys(
        rr_raw
    )

    paired_hr_br_raw = (
        restrict_to_keys(
            hr_br_raw,
            rr_keys,
        )
    )

    if (
        exact_keys(
            paired_hr_br_raw
        )
        != rr_keys
    ):
        raise RuntimeError(
            "Paired datasets do not contain "
            "the same windows"
        )

    paired_hr_br = (
        add_calibration(
            paired_hr_br_raw,
            "hr_br",
        )
    )

    paired_rr = (
        add_calibration(
            rr_raw,
            "hr_br_rr",
        )
    )

    if (
        exact_keys(
            paired_hr_br
        )
        != exact_keys(
            paired_rr
        )
    ):
        raise RuntimeError(
            "Paired datasets differ "
            "after calibration"
        )

    print(
        f"Paired recordings="
        f"{paired_rr['recording'].nunique()}"
    )

    print(
        f"Paired subjects="
        f"{paired_rr['subject'].nunique()}"
    )

    print(
        f"Paired windows="
        f"{len(paired_rr)}"
    )

    print(
        f"Calibration windows="
        f"{int(paired_rr['is_calibration'].sum())}"
    )

    paired_evaluation = select_evaluation_windows(
        paired_rr
    )

    print(
        f"Evaluation windows="
        f"{len(paired_evaluation)}"
    )

    paired_hr_results, _ = evaluate(
        paired_hr_br,
        "paired",
        "hr_br",
    )

    paired_rr_results, _ = evaluate(
        paired_rr,
        "paired",
        "hr_br_rr",
    )

    paired_results = pd.concat(
        [
            paired_hr_results,
            paired_rr_results,
        ],
        ignore_index=True,
    )

    print_results(
        "PAIRED RR ABLATION — EXACT SAME WINDOWS",
        paired_results,
    )


if __name__ == "__main__":
    main()
