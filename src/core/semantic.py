import numpy as np
import pandas as pd
import shap
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from src.core.modeling import (
    add_calibration,
    build_raw_table,
    feature_sets,
    random_forest,
    select_evaluation_windows,
)


FEATURE_SET = "personalized_14"
TOP_K = 5
EPS = 1e-8

STATE_NAMES = {
    0: "Rest",
    1: "Effort",
    2: "Recovery",
}

FEATURE_TO_CONCEPT = {
    "mean_HR": "HR",
    "std_HR": "HR",
    "max_HR": "HR",
    "slope_HR_per_sec": "HR",
    "delta_HR": "HR",
    "mean_HR_delta_cal": "HR",
    "mean_HR_ratio_cal": "HR",
    "mean_BR": "BR",
    "std_BR": "BR",
    "max_BR": "BR",
    "slope_BR_per_sec": "BR",
    "delta_BR": "BR",
    "mean_BR_delta_cal": "BR",
    "mean_BR_ratio_cal": "BR",
}


def shap_matrix(
    model,
    x,
):
    explainer = shap.TreeExplainer(
        model
    )

    values = explainer.shap_values(
        x
    )

    if isinstance(
        values,
        list,
    ):
        return np.stack(
            values,
            axis=2,
        )

    values = np.asarray(
        values
    )

    if values.ndim == 3:
        if values.shape[2] == 3:
            return values

        if values.shape[0] == 3:
            return np.transpose(
                values,
                (
                    1,
                    2,
                    0,
                ),
            )

    raise ValueError(
        f"Unexpected SHAP shape: "
        f"{values.shape}"
    )


def robust_scale(
    frame,
    feature,
):
    values = frame[
        feature
    ].to_numpy(
        dtype=float
    )

    scale = np.std(
        values
    )

    if (
        not np.isfinite(
            scale
        )
        or scale < EPS
    ):
        scale = np.median(
            np.abs(
                values
                - np.median(
                    values
                )
            )
        )

    if (
        not np.isfinite(
            scale
        )
        or scale < EPS
    ):
        scale = 1.0

    return float(
        scale
    )


def build_profiles(
    train,
    features,
):
    global_profiles = {}

    for state in STATE_NAMES:
        part = train[
            train[
                "state"
            ]
            == state
        ]

        global_profiles[
            state
        ] = {
            feature:
            float(
                part[
                    feature
                ].median()
            )
            for feature in features
        }

    sport_profiles = {}

    for sport in sorted(
        train[
            "sport"
        ].unique()
    ):
        sport_part = train[
            train[
                "sport"
            ]
            == sport
        ]

        state_profiles = {}

        for state in STATE_NAMES:
            part = sport_part[
                sport_part[
                    "state"
                ]
                == state
            ]

            if len(part) == 0:
                continue

            state_profiles[
                state
            ] = {
                feature:
                float(
                    part[
                        feature
                    ].median()
                )
                for feature
                in features
            }

        sport_profiles[
            sport
        ] = state_profiles

    global_scale = {
        feature:
        robust_scale(
            train,
            feature,
        )
        for feature in features
    }

    sport_scale = {}

    for sport in sorted(
        train[
            "sport"
        ].unique()
    ):
        part = train[
            train[
                "sport"
            ]
            == sport
        ]

        sport_scale[
            sport
        ] = {
            feature:
            robust_scale(
                part,
                feature,
            )
            for feature in features
        }

    return (
        global_profiles,
        sport_profiles,
        global_scale,
        sport_scale,
    )


def distances_for_feature(
    value,
    feature,
    sport,
    global_profiles,
    sport_profiles,
    global_scale,
    sport_scale,
):
    distances = []

    has_sport = (
        sport
        in sport_profiles
    )

    scale = (
        sport_scale[
            sport
        ][
            feature
        ]
        if sport
        in sport_scale
        else global_scale[
            feature
        ]
    )

    for state in STATE_NAMES:
        if (
            has_sport
            and state
            in sport_profiles[
                sport
            ]
        ):
            reference = (
                sport_profiles[
                    sport
                ][
                    state
                ][
                    feature
                ]
            )
        else:
            reference = (
                global_profiles[
                    state
                ][
                    feature
                ]
            )

        distance = (
            abs(
                value
                - reference
            )
            / max(
                scale,
                EPS,
            )
        )

        distances.append(
            distance
        )

    return np.asarray(
        distances,
        dtype=float,
    )


def similarities(
    distances,
):
    minimum = np.min(
        distances
    )

    scores = np.exp(
        -(
            distances
            - minimum
        )
    )

    denominator = np.sum(
        scores
    )

    if denominator <= 0:
        return np.ones(
            len(
                distances
            )
        ) / len(
            distances
        )

    return (
        scores
        / denominator
    )


def semantic_consistency(
    row,
    predicted_class,
    shap_values,
    features,
    sport,
    profiles,
):
    (
        global_profiles,
        sport_profiles,
        global_scale,
        sport_scale,
    ) = profiles

    positive = np.where(
        shap_values > 0
    )[0]

    if len(
        positive
    ) == 0:
        return (
            np.nan,
            [],
        )

    order = positive[
        np.argsort(
            shap_values[
                positive
            ]
        )[
            ::-1
        ]
    ]

    selected = order[
        :TOP_K
    ]

    weights = np.abs(
        shap_values[
            selected
        ]
    )

    values = []

    details = []

    for index, weight in zip(
        selected,
        weights,
    ):
        feature = features[
            index
        ]

        distance = (
            distances_for_feature(
                float(
                    row[
                        feature
                    ]
                ),
                feature,
                sport,
                global_profiles,
                sport_profiles,
                global_scale,
                sport_scale,
            )
        )

        similarity = similarities(
            distance
        )

        value = float(
            similarity[
                predicted_class
            ]
        )

        values.append(
            value
        )

        details.append(
            (
                feature,
                FEATURE_TO_CONCEPT[
                    feature
                ],
                float(
                    shap_values[
                        index
                    ]
                ),
                value,
            )
        )

    weights_sum = float(
        np.sum(
            weights
        )
    )

    if weights_sum <= EPS:
        return (
            np.nan,
            details,
        )

    consistency = float(
        np.sum(
            weights
            * np.asarray(
                values
            )
        )
        / weights_sum
    )

    return (
        consistency,
        details,
    )


def main():
    print(
        "Loading HR+BR personalized dataset..."
    )

    raw = build_raw_table(
        "hr_br"
    )

    df = add_calibration(
        raw,
        "hr_br",
    )

    evaluation = select_evaluation_windows(
        df
    )

    features = feature_sets(
        "hr_br"
    )[
        FEATURE_SET
    ]

    x = evaluation[
        features
    ].to_numpy(
        dtype=float
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

    pred = np.full(
        len(
            evaluation
        ),
        -1,
        dtype=int,
    )

    confidence = np.full(
        len(
            evaluation
        ),
        np.nan,
        dtype=float,
    )

    consistency = np.full(
        len(
            evaluation
        ),
        np.nan,
        dtype=float,
    )

    top_features = [
        None
        for _
        in range(
            len(
                evaluation
            )
        )
    ]

    splitter = LeaveOneGroupOut()

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
        print(
            f"fold={fold:02d} "
            f"test={groups[test_idx[0]]} "
            f"windows={len(test_idx)}"
        )

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

        fold_proba = model.predict_proba(
            x[
                test_idx
            ]
        )

        fold_shap = shap_matrix(
            model,
            x[
                test_idx
            ],
        )

        train_frame = (
            evaluation.iloc[
                train_idx
            ]
        )

        profiles = build_profiles(
            train_frame,
            features,
        )

        pred[
            test_idx
        ] = fold_pred

        for local_index, global_index in enumerate(
            test_idx
        ):
            predicted_class = int(
                fold_pred[
                    local_index
                ]
            )

            class_position = int(
                np.where(
                    model.classes_
                    == predicted_class
                )[0][0]
            )

            confidence[
                global_index
            ] = float(
                fold_proba[
                    local_index,
                    class_position,
                ]
            )

            class_shap = fold_shap[
                local_index,
                :,
                class_position,
            ]

            score, details = (
                semantic_consistency(
                    evaluation.iloc[
                        global_index
                    ],
                    predicted_class,
                    class_shap,
                    features,
                    sports[
                        global_index
                    ],
                    profiles,
                )
            )

            consistency[
                global_index
            ] = score

            top_features[
                global_index
            ] = details

    correct = (
        pred
        == y
    )

    valid = np.isfinite(
        consistency
    )

    print("")
    print("=" * 100)
    print(
        "OOF XAI SEMANTIC V2"
    )
    print("=" * 100)

    print(
        f"windows={len(y)}"
    )

    print(
        f"accuracy="
        f"{accuracy_score(y, pred):.4f}"
    )

    print(
        f"valid_consistency="
        f"{int(valid.sum())}"
    )

    print(
        f"missing_consistency="
        f"{int((~valid).sum())}"
    )

    print("")
    print(
        "CONFIDENCE"
    )

    print(
        f"correct_mean="
        f"{np.mean(confidence[correct]):.4f}"
    )

    print(
        f"error_mean="
        f"{np.mean(confidence[~correct]):.4f}"
    )

    print(
        f"correct_median="
        f"{np.median(confidence[correct]):.4f}"
    )

    print(
        f"error_median="
        f"{np.median(confidence[~correct]):.4f}"
    )

    print("")
    print(
        "SEMANTIC CONSISTENCY"
    )

    print(
        f"correct_mean="
        f"{np.mean(consistency[correct & valid]):.4f}"
    )

    print(
        f"error_mean="
        f"{np.mean(consistency[(~correct) & valid]):.4f}"
    )

    print(
        f"correct_median="
        f"{np.median(consistency[correct & valid]):.4f}"
    )

    print(
        f"error_median="
        f"{np.median(consistency[(~correct) & valid]):.4f}"
    )

    reliability = np.sqrt(
        confidence
        * consistency
    )

    valid_r = np.isfinite(
        reliability
    )

    print("")
    print(
        "RELIABILITY"
    )

    print(
        f"correct_mean="
        f"{np.mean(reliability[correct & valid_r]):.4f}"
    )

    print(
        f"error_mean="
        f"{np.mean(reliability[(~correct) & valid_r]):.4f}"
    )

    print(
        f"correct_median="
        f"{np.median(reliability[correct & valid_r]):.4f}"
    )

    print(
        f"error_median="
        f"{np.median(reliability[(~correct) & valid_r]):.4f}"
    )

    print("")
    print("=" * 100)
    print(
        "BY PREDICTED STATE"
    )
    print("=" * 100)

    for state in STATE_NAMES:
        mask = (
            pred
            == state
        )

        valid_mask = (
            mask
            & valid
        )

        print("")
        print(
            STATE_NAMES[
                state
            ]
        )

        print(
            f"n="
            f"{int(mask.sum())}"
        )

        print(
            f"accuracy="
            f"{np.mean(correct[mask]):.4f}"
        )

        print(
            f"confidence_mean="
            f"{np.mean(confidence[mask]):.4f}"
        )

        print(
            f"consistency_mean="
            f"{np.mean(consistency[valid_mask]):.4f}"
        )

        print(
            f"consistency_min="
            f"{np.min(consistency[valid_mask]):.4f}"
        )

        print(
            f"consistency_median="
            f"{np.median(consistency[valid_mask]):.4f}"
        )

        print(
            f"consistency_max="
            f"{np.max(consistency[valid_mask]):.4f}"
        )

        print(
            f"reliability_mean="
            f"{np.mean(reliability[valid_mask]):.4f}"
        )

    print("")
    print("=" * 100)
    print(
        "BY SPORT"
    )
    print("=" * 100)

    for sport in sorted(
        set(
            sports
        )
    ):
        mask = (
            sports
            == sport
        )

        valid_mask = (
            mask
            & valid
        )

        print(
            f"{sport}: "
            f"n={int(mask.sum())} "
            f"acc={np.mean(correct[mask]):.4f} "
            f"conf={np.mean(confidence[mask]):.4f} "
            f"consistency="
            f"{np.mean(consistency[valid_mask]):.4f} "
            f"reliability="
            f"{np.mean(reliability[valid_mask]):.4f}"
        )

    print("")
    print("=" * 100)
    print(
        "CONSISTENCY QUANTILES"
    )
    print("=" * 100)

    for label, mask in [
        (
            "correct",
            correct & valid,
        ),
        (
            "error",
            (~correct)
            & valid,
        ),
    ]:
        values = consistency[
            mask
        ]

        print(
            f"{label}: "
            f"q05={np.quantile(values, 0.05):.4f} "
            f"q25={np.quantile(values, 0.25):.4f} "
            f"q50={np.quantile(values, 0.50):.4f} "
            f"q75={np.quantile(values, 0.75):.4f} "
            f"q95={np.quantile(values, 0.95):.4f}"
        )

    output_dir = Path(
        "results/baseline"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = evaluation[
        [
            "recording",
            "subject",
            "sport",
            "state",
            "window_start_sample",
            "window_end_sample",
        ]
    ].copy()

    output[
        "prediction"
    ] = pred

    output[
        "correct"
    ] = correct.astype(
        int
    )

    output[
        "confidence"
    ] = confidence

    output[
        "consistency"
    ] = consistency

    output[
        "reliability"
    ] = reliability

    output[
        "true_state_name"
    ] = [
        STATE_NAMES[
            int(value)
        ]
        for value in y
    ]

    output[
        "predicted_state_name"
    ] = [
        STATE_NAMES[
            int(value)
        ]
        for value in pred
    ]

    output.to_csv(
        output_dir
        / "xai_oof_predictions.csv",
        index=False,
    )

    print("")
    print(
        "saved="
        "results/baseline/"
        "xai_oof_predictions.csv"
    )



if __name__ == "__main__":
    main()
