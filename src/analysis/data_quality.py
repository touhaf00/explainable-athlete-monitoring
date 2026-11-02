from collections import Counter, defaultdict

import numpy as np

from src.core.dataset import (
    LABEL_NAMES,
    load_dataset_v2,
)


EXPECTED = {
    "hr_br": {
        "recordings": 42,
        "subjects": 38,
        "channels": 2,
        "features": 6,
    },
    "hr_br_rr": {
        "recordings": 31,
        "subjects": 27,
        "channels": 3,
        "features": 9,
    },
}


def check_mode(mode):
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
    ) = load_dataset_v2(mode)

    print("=" * 80)
    print(f"DATA QUALITY V2 — {mode}")
    print("=" * 80)

    expected = EXPECTED[mode]

    recordings = len(
        set(
            item["recording"]
            for item in metadata
        )
    )

    subjects = len(
        set(groups)
    )

    print(f"recordings={recordings}")
    print(f"subjects={subjects}")
    print(f"windows={len(y)}")
    print(f"sequence_shape={x_seq.shape}")
    print(f"statistical_shape={x_stat.shape}")

    print("")

    print(
        "class_distribution="
        + str(
            dict(
                Counter(
                    LABEL_NAMES[
                        int(value)
                    ]
                    for value in y
                )
            )
        )
    )

    print(
        "windows_per_sport="
        + str(
            dict(
                Counter(
                    sports
                )
            )
        )
    )

    if recordings != expected[
        "recordings"
    ]:
        raise SystemExit(
            f"ERROR recordings: "
            f"{recordings} != "
            f"{expected['recordings']}"
        )

    if subjects != expected[
        "subjects"
    ]:
        raise SystemExit(
            f"ERROR subjects: "
            f"{subjects} != "
            f"{expected['subjects']}"
        )

    if x_seq.ndim != 3:
        raise SystemExit(
            "ERROR sequence dimensions"
        )

    if x_seq.shape[1] != 50:
        raise SystemExit(
            "ERROR window size"
        )

    if x_seq.shape[2] != expected[
        "channels"
    ]:
        raise SystemExit(
            "ERROR number of channels"
        )

    if x_stat.shape[1] != expected[
        "features"
    ]:
        raise SystemExit(
            "ERROR number of statistical features"
        )

    print("")
    print("Finite values")

    sequence_nonfinite = int(
        np.size(x_seq)
        - np.isfinite(
            x_seq
        ).sum()
    )

    statistical_nonfinite = int(
        np.size(x_stat)
        - np.isfinite(
            x_stat
        ).sum()
    )

    print(
        f"sequence_nonfinite="
        f"{sequence_nonfinite}"
    )

    print(
        f"statistical_nonfinite="
        f"{statistical_nonfinite}"
    )

    if (
        sequence_nonfinite
        or statistical_nonfinite
    ):
        raise SystemExit(
            "ERROR non-finite values remain"
        )

    print("")
    print("Window identity")

    positions = Counter(
        (
            item["recording"],
            item["window_start_sample"],
        )
        for item in metadata
    )

    duplicate_positions = sum(
        count > 1
        for count in positions.values()
    )

    print(
        f"duplicate_window_positions="
        f"{duplicate_positions}"
    )

    if duplicate_positions:
        raise SystemExit(
            "ERROR duplicate window positions"
        )

    exact_sequences = defaultdict(
        set
    )

    for sequence, label, item in zip(
        x_seq,
        y,
        metadata,
    ):
        key = (
            item["recording"],
            sequence.tobytes(),
        )

        exact_sequences[
            key
        ].add(
            int(label)
        )

    contradictory_sequences = sum(
        len(values) > 1
        for values
        in exact_sequences.values()
    )

    print(
        "contradictory_exact_sequences="
        f"{contradictory_sequences}"
    )

    if contradictory_sequences:
        raise SystemExit(
            "ERROR contradictory exact sequences"
        )

    statistical_labels = defaultdict(
        set
    )

    for features, label, item in zip(
        x_stat,
        y,
        metadata,
    ):
        key = (
            item["recording"],
            features.tobytes(),
        )

        statistical_labels[
            key
        ].add(
            int(label)
        )

    same_stats_multiple_labels = sum(
        len(values) > 1
        for values
        in statistical_labels.values()
    )

    print(
        "same_stats_multiple_labels="
        f"{same_stats_multiple_labels}"
    )

    print("")
    print("Chronology")

    by_recording = defaultdict(
        list
    )

    for item in metadata:
        by_recording[
            item["recording"]
        ].append(
            (
                item[
                    "segment_start_sec"
                ],
                item[
                    "window_start_sample"
                ],
            )
        )

    chronological = True

    for values in (
        by_recording.values()
    ):
        if any(
            values[index]
            > values[index + 1]
            for index in range(
                len(values) - 1
            )
        ):
            chronological = False
            break

    print(
        f"chronological_windows="
        f"{chronological}"
    )

    if not chronological:
        raise SystemExit(
            "ERROR chronology"
        )

    print("")
    print("Subject state coverage")

    states_by_subject = defaultdict(
        set
    )

    windows_by_subject_state = Counter()

    for label, subject in zip(
        y,
        groups,
    ):
        state = LABEL_NAMES[
            int(label)
        ]

        states_by_subject[
            subject
        ].add(
            state
        )

        windows_by_subject_state[
            (
                subject,
                state,
            )
        ] += 1

    required_states = set(
        LABEL_NAMES.values()
    )

    missing_states = {
        subject:
        sorted(
            required_states
            - states
        )
        for subject, states
        in states_by_subject.items()
        if states
        != required_states
    }

    print(
        "subjects_missing_states="
        + str(
            missing_states
        )
    )

    rest_counts = np.array(
        [
            windows_by_subject_state[
                (
                    subject,
                    "S0_Rest",
                )
            ]
            for subject
            in sorted(
                states_by_subject
            )
        ],
        dtype=int,
    )

    recovery_counts = np.array(
        [
            windows_by_subject_state[
                (
                    subject,
                    "S2_Recovery",
                )
            ]
            for subject
            in sorted(
                states_by_subject
            )
        ],
        dtype=int,
    )

    print(
        "rest_windows_per_subject="
        f"min:{rest_counts.min()} "
        f"median:{np.median(rest_counts):.1f} "
        f"max:{rest_counts.max()}"
    )

    print(
        "recovery_windows_per_subject="
        f"min:{recovery_counts.min()} "
        f"median:{np.median(recovery_counts):.1f} "
        f"max:{recovery_counts.max()}"
    )

    print("")
    print("State distribution by sport")

    state_sport = Counter(
        (
            str(sport),
            LABEL_NAMES[
                int(label)
            ],
        )
        for sport, label
        in zip(
            sports,
            y,
        )
    )

    for sport in sorted(
        set(
            sports
        )
    ):
        values = {
            state:
            state_sport[
                (
                    sport,
                    state,
                )
            ]
            for state
            in LABEL_NAMES.values()
        }

        print(
            f"{sport}={values}"
        )

    print("")
    print("Truncation")

    truncated = {}

    for item in metadata:
        seconds = float(
            item[
                "truncated_seconds"
            ]
        )

        if seconds <= 0:
            continue

        recording = item[
            "recording"
        ]

        truncated[
            recording
        ] = max(
            truncated.get(
                recording,
                0.0,
            ),
            seconds,
        )

    print(
        "truncated_seconds="
        + str(
            {
                recording:
                round(
                    seconds,
                    1,
                )
                for recording, seconds
                in truncated.items()
            }
        )
    )

    print(
        "ignored_segments="
        + str(
            dict(
                ignored
            )
        )
    )

    if mode == "hr_br_rr":
        print("")
        print("RR quality")

        minimum_coverages = {}

        for recording in used:
            coverage = rr_quality[
                recording
            ]

            minimum_coverages[
                recording
            ] = min(
                coverage.values()
            )

        values = np.array(
            list(
                minimum_coverages.values()
            ),
            dtype=float,
        )

        print(
            "minimum_state_RR_coverage="
            f"min:{values.min():.4f} "
            f"median:{np.median(values):.4f} "
            f"max:{values.max():.4f}"
        )

        if np.any(
            values < 0.80
        ):
            raise SystemExit(
                "ERROR RR quality threshold"
            )

    print("")
    print(
        f"DATA QUALITY PASSED — "
        f"{mode}"
    )

    print("")


def main():
    check_mode(
        "hr_br"
    )

    check_mode(
        "hr_br_rr"
    )


if __name__ == "__main__":
    main()
