import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from src.core.modeling import (
    add_calibration,
    build_raw_table,
    feature_sets,
    random_forest,
    select_evaluation_windows,
)


FEATURE_SET = "personalized_14"

STATE_NAMES = {
    0: "Rest",
    1: "Effort",
    2: "Recovery",
}


def metrics_block(
    y_true,
    y_pred,
):
    recalls = recall_score(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
        ],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy":
        accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy":
        balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "macro_f1":
        f1_score(
            y_true,
            y_pred,
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


def main():
    print(
        "Loading primary HR+BR dataset..."
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
        len(y),
        -1,
        dtype=int,
    )

    probability = np.full(
        (
            len(y),
            3,
        ),
        np.nan,
        dtype=float,
    )

    splitter = LeaveOneGroupOut()

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
        subject = str(
            groups[
                test_idx[0]
            ]
        )

        subject_sports = sorted(
            set(
                sports[
                    test_idx
                ]
            )
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

        pred[
            test_idx
        ] = fold_pred

        for class_position, class_id in enumerate(
            model.classes_
        ):
            probability[
                test_idx,
                int(
                    class_id
                ),
            ] = fold_proba[
                :,
                class_position,
            ]

        metrics = metrics_block(
            y[
                test_idx
            ],
            fold_pred,
        )

        true_counts = {
            state:
            int(
                np.sum(
                    y[
                        test_idx
                    ]
                    == state_id
                )
            )
            for state_id, state
            in STATE_NAMES.items()
        }

        fold_rows.append(
            {
                "fold":
                fold,

                "subject":
                subject,

                "sport":
                "+".join(
                    subject_sports
                ),

                "n_windows":
                len(
                    test_idx
                ),

                "n_rest":
                true_counts[
                    "Rest"
                ],

                "n_effort":
                true_counts[
                    "Effort"
                ],

                "n_recovery":
                true_counts[
                    "Recovery"
                ],

                **metrics,
            }
        )

    if np.any(
        pred < 0
    ):
        raise RuntimeError(
            "Some windows were not predicted"
        )

    fold_df = pd.DataFrame(
        fold_rows
    )

    overall = metrics_block(
        y,
        pred,
    )

    print("")
    print("=" * 110)
    print(
        "OVERALL OOF — PERSONALIZED_14"
    )
    print("=" * 110)

    print(
        f"windows={len(y)}"
    )

    print(
        f"subjects={len(set(groups))}"
    )

    print(
        f"accuracy="
        f"{overall['accuracy']:.4f}"
    )

    print(
        f"balanced_accuracy="
        f"{overall['balanced_accuracy']:.4f}"
    )

    print(
        f"macro_f1="
        f"{overall['macro_f1']:.4f}"
    )

    print(
        f"recall_rest="
        f"{overall['recall_rest']:.4f}"
    )

    print(
        f"recall_effort="
        f"{overall['recall_effort']:.4f}"
    )

    print(
        f"recall_recovery="
        f"{overall['recall_recovery']:.4f}"
    )

    print("")
    print("=" * 110)
    print(
        "PERFORMANCE BY SPORT"
    )
    print("=" * 110)

    sport_rows = []

    for sport in sorted(
        set(
            sports
        )
    ):
        mask = (
            sports
            == sport
        )

        metrics = metrics_block(
            y[
                mask
            ],
            pred[
                mask
            ],
        )

        sport_rows.append(
            {
                "sport":
                sport,

                "subjects":
                len(
                    set(
                        groups[
                            mask
                        ]
                    )
                ),

                "windows":
                int(
                    mask.sum()
                ),

                **metrics,
            }
        )

    sport_df = pd.DataFrame(
        sport_rows
    )

    sport_columns = [
        "sport",
        "subjects",
        "windows",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_rest",
        "recall_effort",
        "recall_recovery",
    ]

    sport_display = sport_df[
        sport_columns
    ].copy()

    numeric = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_rest",
        "recall_effort",
        "recall_recovery",
    ]

    sport_display[
        numeric
    ] = sport_display[
        numeric
    ].round(
        4
    )

    print(
        sport_display.to_string(
            index=False
        )
    )

    print("")
    print("=" * 110)
    print(
        "PERFORMANCE BY SUBJECT — WORST TO BEST"
    )
    print("=" * 110)

    subject_columns = [
        "subject",
        "sport",
        "n_windows",
        "n_rest",
        "n_effort",
        "n_recovery",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_rest",
        "recall_effort",
        "recall_recovery",
    ]

    subject_display = (
        fold_df[
            subject_columns
        ]
        .sort_values(
            [
                "balanced_accuracy",
                "accuracy",
            ],
            ascending=True,
        )
        .copy()
    )

    subject_display[
        numeric
    ] = subject_display[
        numeric
    ].round(
        4
    )

    print(
        subject_display.to_string(
            index=False
        )
    )

    print("")
    print("=" * 110)
    print(
        "SUBJECT DISTRIBUTION"
    )
    print("=" * 110)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "recall_recovery",
    ]:
        values = fold_df[
            metric
        ].to_numpy(
            dtype=float
        )

        print(
            f"{metric}: "
            f"min={values.min():.4f} "
            f"q25={np.quantile(values, 0.25):.4f} "
            f"median={np.median(values):.4f} "
            f"q75={np.quantile(values, 0.75):.4f} "
            f"max={values.max():.4f}"
        )

    print("")
    print("=" * 110)
    print(
        "CONFUSION MATRIX — COUNTS"
    )
    print("=" * 110)

    cm = confusion_matrix(
        y,
        pred,
        labels=[
            0,
            1,
            2,
        ],
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "True Rest",
            "True Effort",
            "True Recovery",
        ],
        columns=[
            "Pred Rest",
            "Pred Effort",
            "Pred Recovery",
        ],
    )

    print(
        cm_df.to_string()
    )

    print("")
    print("=" * 110)
    print(
        "CONFUSION MATRIX — ROW NORMALIZED"
    )
    print("=" * 110)

    cm_normalized = (
        cm
        / cm.sum(
            axis=1,
            keepdims=True,
        )
    )

    cm_normalized_df = pd.DataFrame(
        cm_normalized,
        index=[
            "True Rest",
            "True Effort",
            "True Recovery",
        ],
        columns=[
            "Pred Rest",
            "Pred Effort",
            "Pred Recovery",
        ],
    )

    print(
        cm_normalized_df.round(
            4
        ).to_string()
    )

    confidence = probability[
        np.arange(
            len(pred)
        ),
        pred,
    ]

    correct = (
        pred
        == y
    )

    print("")
    print("=" * 110)
    print(
        "PREDICTIVE CONFIDENCE"
    )
    print("=" * 110)

    print(
        "correct_predictions="
        f"{int(correct.sum())}"
    )

    print(
        "errors="
        f"{int((~correct).sum())}"
    )

    print(
        "mean_confidence_correct="
        f"{np.mean(confidence[correct]):.4f}"
    )

    print(
        "mean_confidence_error="
        f"{np.mean(confidence[~correct]):.4f}"
    )

    print(
        "median_confidence_correct="
        f"{np.median(confidence[correct]):.4f}"
    )

    print(
        "median_confidence_error="
        f"{np.median(confidence[~correct]):.4f}"
    )

    print("")
    print("=" * 110)
    print(
        "RECOVERY ERRORS"
    )
    print("=" * 110)

    recovery_mask = (
        y
        == 2
    )

    recovery_errors = CounterLike(
        pred[
            recovery_mask
        ]
    )

    print(
        f"true_recovery_pred_rest="
        f"{recovery_errors.get(0, 0)}"
    )

    print(
        f"true_recovery_pred_effort="
        f"{recovery_errors.get(1, 0)}"
    )

    print(
        f"true_recovery_pred_recovery="
        f"{recovery_errors.get(2, 0)}"
    )


def CounterLike(
    values,
):
    result = {}

    for value in values:
        value = int(
            value
        )

        result[
            value
        ] = (
            result.get(
                value,
                0,
            )
            + 1
        )

    return result


if __name__ == "__main__":
    main()
