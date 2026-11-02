from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from explainability_kg import (
    FEATURES,
    ID_TO_STATE,
    OUTPUT_DIR,
    build_state_profiles,
    hitl_flag,
    kg_consistency,
    load_data,
    make_model,
    make_text_explanation,
)


OOF_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "local_explanations_oof.csv",
)

FOLD_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "oof_fold_summary.csv",
)

SHAP_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "oof_shap_importance.csv",
)


def normalize_shap_output(
    raw_values: Any,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    if isinstance(raw_values, list):
        values = np.asarray(
            raw_values,
            dtype=float,
        )
    else:
        values = np.asarray(
            raw_values,
            dtype=float,
        )

    if values.ndim == 2:
        if n_classes != 2:
            raise ValueError(
                "A two-dimensional SHAP output was returned for a "
                "non-binary model."
            )

        negative_class = -values
        positive_class = values

        return np.stack(
            [
                negative_class,
                positive_class,
            ],
            axis=0,
        )

    if values.ndim != 3:
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    if values.shape == (
        n_classes,
        n_samples,
        n_features,
    ):
        return values

    if values.shape == (
        n_samples,
        n_features,
        n_classes,
    ):
        return np.moveaxis(
            values,
            -1,
            0,
        )

    if values.shape == (
        n_samples,
        n_classes,
        n_features,
    ):
        return np.moveaxis(
            values,
            1,
            0,
        )

    raise ValueError(
        f"Unable to interpret SHAP shape: {values.shape}"
    )


def top_supporting_features(
    row: pd.Series,
    class_shap_values: np.ndarray,
    predicted_state: str,
    sport_profiles: pd.DataFrame,
    global_profiles: pd.DataFrame,
    sport_scales: pd.DataFrame,
    global_scales: pd.Series,
) -> tuple[float, list[dict[str, Any]]]:
    consistency, details = kg_consistency(
        row=row,
        predicted_state=predicted_state,
        class_shap_values=class_shap_values,
        sport_profiles=sport_profiles,
        global_profiles=global_profiles,
        sport_scales=sport_scales,
        global_scales=global_scales,
    )

    selected = [
        dict(item)
        for item in details[:3]
    ]

    while len(selected) < 3:
        selected.append({
            "feature": "",
            "value": np.nan,
            "shap_value": np.nan,
            "similarity": np.nan,
            "concept": "",
        })

    return consistency, selected


def build_oof_explanations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[FEATURES].to_numpy(
        dtype=float
    )
    y = df["label"].to_numpy(
        dtype=int
    )
    groups = df["subject"].to_numpy(
        dtype=str
    )

    splitter = LeaveOneGroupOut()

    explanation_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    shap_rows: list[dict[str, Any]] = []

    folds = list(
        splitter.split(
            x,
            y,
            groups,
        )
    )

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        folds,
        start=1,
    ):
        train_df = df.iloc[
            train_index
        ].copy()

        test_df = df.iloc[
            test_index
        ].copy()

        held_out_group = str(
            groups[test_index[0]]
        )

        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            x[train_index]
        )

        x_test = scaler.transform(
            x[test_index]
        )

        model = make_model(
            n_estimators=300
        )

        model.fit(
            x_train,
            y[train_index],
        )

        predictions = model.predict(
            x_test
        )

        probabilities = model.predict_proba(
            x_test
        )

        fold_accuracy = accuracy_score(
            y[test_index],
            predictions,
        )

        (
            sport_profiles,
            global_profiles,
            sport_scales,
            global_scales,
        ) = build_state_profiles(
            train_df
        )

        explainer = shap.TreeExplainer(
            model
        )

        raw_shap_values = explainer.shap_values(
            x_test
        )

        shap_values = normalize_shap_output(
            raw_values=raw_shap_values,
            n_samples=len(test_index),
            n_features=len(FEATURES),
            n_classes=len(model.classes_),
        )

        class_positions = {
            int(class_id): position
            for position, class_id in enumerate(
                model.classes_
            )
        }

        for local_index, (
            dataframe_index,
            predicted_id,
        ) in enumerate(
            zip(
                test_index,
                predictions,
            )
        ):
            row = df.iloc[
                dataframe_index
            ]

            predicted_id = int(
                predicted_id
            )

            predicted_state = ID_TO_STATE[
                predicted_id
            ]

            class_position = class_positions[
                predicted_id
            ]

            confidence = float(
                probabilities[
                    local_index,
                    class_position,
                ]
            )

            class_shap_values = np.asarray(
                shap_values[
                    class_position,
                    local_index,
                ],
                dtype=float,
            )

            (
                consistency,
                supporting_features,
            ) = top_supporting_features(
                row=row,
                class_shap_values=class_shap_values,
                predicted_state=predicted_state,
                sport_profiles=sport_profiles,
                global_profiles=global_profiles,
                sport_scales=sport_scales,
                global_scales=global_scales,
            )

            explanation = make_text_explanation(
                predicted_state=predicted_state,
                confidence=confidence,
                supporting_features=supporting_features,
                consistency=consistency,
            )

            explanation_rows.append({
                "window_id": int(
                    row["window_id"]
                ),
                "sport": str(
                    row["sport"]
                ),
                "subject": str(
                    row["subject"]
                ),
                "held_out_group": held_out_group,
                "fold_number": fold_number,
                "evaluation_mode": "LOSO_out_of_fold",
                "true_state": str(
                    row["true_state"]
                ),
                "prediction": predicted_state,
                "correct": int(
                    str(row["true_state"])
                    == predicted_state
                ),
                "confidence": round(
                    confidence,
                    6,
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
                    6,
                ),
                "hitl_flag": hitl_flag(
                    confidence,
                    consistency,
                ),
                "explanation": explanation,
            })

            for feature_index, feature_name in enumerate(
                FEATURES
            ):
                shap_rows.append({
                    "fold_number": fold_number,
                    "held_out_group": held_out_group,
                    "window_id": int(
                        row["window_id"]
                    ),
                    "prediction": predicted_state,
                    "feature": feature_name,
                    "shap_value": float(
                        class_shap_values[
                            feature_index
                        ]
                    ),
                    "absolute_shap_value": abs(
                        float(
                            class_shap_values[
                                feature_index
                            ]
                        )
                    ),
                })

        fold_rows.append({
            "fold_number": fold_number,
            "held_out_group": held_out_group,
            "sport": str(
                test_df["sport"].iloc[0]
            ),
            "n_train": len(train_index),
            "n_test": len(test_index),
            "accuracy": float(
                fold_accuracy
            ),
        })

        print(
            f"Fold {fold_number:02d}/{len(folds)} | "
            f"{held_out_group:<24} | "
            f"n={len(test_index):4d} | "
            f"accuracy={fold_accuracy:.3f}"
        )

    explanations = pd.DataFrame(
        explanation_rows
    ).sort_values(
        "window_id"
    ).reset_index(
        drop=True
    )

    fold_summary = pd.DataFrame(
        fold_rows
    )

    shap_details = pd.DataFrame(
        shap_rows
    )

    return (
        explanations,
        fold_summary,
        shap_details,
    )


def save_results(
    explanations: pd.DataFrame,
    fold_summary: pd.DataFrame,
    shap_details: pd.DataFrame,
) -> None:
    explanations.to_csv(
        OOF_OUTPUT_FILE,
        index=False,
    )

    fold_summary.to_csv(
        FOLD_OUTPUT_FILE,
        index=False,
    )

    global_shap = (
        shap_details
        .groupby(
            "feature",
            as_index=False,
        )["absolute_shap_value"]
        .mean()
        .rename(
            columns={
                "absolute_shap_value": "mean_absolute_shap",
            }
        )
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
    )

    global_shap.to_csv(
        SHAP_OUTPUT_FILE,
        index=False,
    )


def print_summary(
    explanations: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> None:
    overall_accuracy = float(
        explanations["correct"].mean()
    )

    hitl_summary = (
        explanations
        .groupby(
            "hitl_flag"
        )
        .agg(
            n_windows=(
                "correct",
                "size",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
            mean_confidence=(
                "confidence",
                "mean",
            ),
            mean_consistency=(
                "kg_consistency",
                "mean",
            ),
        )
        .reindex(
            [
                "accept",
                "check",
                "review",
            ]
        )
    )

    print("")
    print("LOSO out-of-fold explanation summary")
    print(
        f"  Overall accuracy: "
        f"{overall_accuracy:.3f}"
    )
    print(
        f"  Mean fold accuracy: "
        f"{fold_summary['accuracy'].mean():.3f} "
        f"+/- {fold_summary['accuracy'].std(ddof=0):.3f}"
    )
    print("")
    print(
        hitl_summary
        .round(4)
        .to_string()
    )


def main() -> None:
    print("LOSO out-of-fold explanations")
    print(
        "Each window is predicted by a model that was trained "
        "without its subject."
    )
    print("")

    df = load_data()

    (
        explanations,
        fold_summary,
        shap_details,
    ) = build_oof_explanations(
        df
    )

    save_results(
        explanations=explanations,
        fold_summary=fold_summary,
        shap_details=shap_details,
    )

    print_summary(
        explanations=explanations,
        fold_summary=fold_summary,
    )

    print("")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("  local_explanations_oof.csv")
    print("  oof_fold_summary.csv")
    print("  oof_shap_importance.csv")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
