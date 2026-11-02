from collections import Counter
from pathlib import Path
import argparse
import re

import numpy as np
from scipy.io import loadmat


BASE_DIR = Path(__file__).resolve().parents[2]
SPORTDB_ROOT = BASE_DIR / "SportDB2"

SPORTS = {
    "AMF",
    "CYC",
    "RUN",
    "SOC",
}

SIGNAL_RATE_HZ = 1.0
WINDOW_SIZE = 50
STEP_SIZE = 25
RR_MIN_STATE_COVERAGE = 0.80

LABEL_NAMES = {
    0: "S0_Rest",
    1: "S1_Effort",
    2: "S2_Recovery",
}

LABEL_TO_ID = {
    value: key
    for key, value in LABEL_NAMES.items()
}

SPORT_LABELS = {
    "AMF": {
        "resting": "S0_Rest",
        "recovery": "S2_Recovery",
        "mobility": "S1_Effort",
        "linear_speed": "S1_Effort",
        "kicking": "S1_Effort",
        "kicking_game": "S1_Effort",
        "team": "S1_Effort",
        "field_goal": "S1_Effort",
        "offense": "S1_Effort",
        "defense": "S1_Effort",
        "2min_game_situation": "S1_Effort",
        "team_defense": "S1_Effort",
        "team_offense": "S1_Effort",
        "7vs7": "S1_Effort",
        "oline_vs_dline": "S1_Effort",
        "technical_exercise": "S1_Effort",
    },
    "CYC": {
        "resting": "S0_Rest",
        "exercise": "S1_Effort",
        "recovery": "S2_Recovery",
    },
    "RUN": {
        "resting": "S0_Rest",
        "i phase": "S1_Effort",
        "ii phase": "S1_Effort",
        "iii phase": "S1_Effort",
        "iv phase": "S1_Effort",
        "recovery": "S2_Recovery",
    },
    "SOC": {
        "resting": "S0_Rest",
        "i phase": "S1_Effort",
        "ii phase": "S1_Effort",
        "iii phase": "S1_Effort",
        "iv phase": "S1_Effort",
        "recovery": "S2_Recovery",
    },
}

PATTERN = re.compile(
    r"([A-Za-z0-9\s\-_+()]+?)\s*:\s*from\s+"
    r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s+to\s+"
    r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",
    re.IGNORECASE,
)


def hms_to_seconds(value):
    hours, minutes, seconds = value.split(":")

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


def normalize_label(value):
    return re.sub(
        r"\s+",
        " ",
        value.strip().lower(),
    )


def parse_annotations(path):
    text = path.read_text(
        encoding="latin-1",
        errors="replace",
    )

    segments = []

    for match in PATTERN.finditer(text):
        segments.append(
            {
                "raw_label": match.group(1).strip(),
                "normalized_label": normalize_label(
                    match.group(1)
                ),
                "start": hms_to_seconds(
                    match.group(2)
                ),
                "end": hms_to_seconds(
                    match.group(3)
                ),
            }
        )

    segments.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )

    if not segments:
        return []

    origin = min(
        item["start"]
        for item in segments
    )

    for segment in segments:
        segment["relative_start"] = (
            segment["start"]
            - origin
        )

        segment["relative_end"] = (
            segment["end"]
            - origin
        )

    return segments


def subject_and_session(data_path):
    relative = data_path.relative_to(
        SPORTDB_ROOT
    )

    sport = relative.parts[0]

    subject = (
        sport
        + "/"
        + "/".join(
            relative.parts[1:-2]
        )
    )

    session = relative.parts[-2]

    recording = str(
        relative.parent
    )

    return subject, session, recording


def mapped_segments(
    sport,
    segments,
    signal_length,
):
    result = []

    for index, segment in enumerate(
        segments
    ):
        state = SPORT_LABELS[
            sport
        ].get(
            segment[
                "normalized_label"
            ]
        )

        if state is None:
            continue

        original_start = float(
            segment[
                "relative_start"
            ]
        )

        original_end = float(
            segment[
                "relative_end"
            ]
        )

        clipped_start = max(
            0.0,
            original_start,
        )

        clipped_end = min(
            float(signal_length),
            original_end,
        )

        start_index = int(
            np.ceil(
                clipped_start
            )
        )

        end_index = int(
            np.floor(
                clipped_end
            )
        )

        if (
            end_index
            - start_index
            < WINDOW_SIZE
        ):
            continue

        result.append(
            {
                "segment_index": index,
                "state": state,
                "raw_label":
                segment[
                    "raw_label"
                ],
                "original_start":
                original_start,
                "original_end":
                original_end,
                "clipped_start":
                clipped_start,
                "clipped_end":
                clipped_end,
                "start_index":
                start_index,
                "end_index":
                end_index,
                "truncated_seconds":
                max(
                    0.0,
                    original_end
                    - float(
                        signal_length
                    ),
                ),
            }
        )

    return result


def rr_state_coverage(
    rr,
    segments,
):
    finite = np.isfinite(
        rr
    )

    totals = Counter()
    valid = Counter()

    for segment in segments:
        state = segment[
            "state"
        ]

        for start in range(
            segment[
                "start_index"
            ],
            segment[
                "end_index"
            ]
            - WINDOW_SIZE
            + 1,
            STEP_SIZE,
        ):
            totals[state] += 1

            if finite[
                start:
                start + WINDOW_SIZE
            ].all():
                valid[state] += 1

    coverage = {}

    for state in LABEL_TO_ID:
        total = totals[
            state
        ]

        coverage[state] = (
            valid[state] / total
            if total
            else 0.0
        )

    return coverage


def statistical_features(
    sequence,
    mode,
):
    hr = sequence[:, 0]
    br = sequence[:, 1]

    values = [
        np.mean(hr),
        np.std(hr),
        np.max(hr),
        np.mean(br),
        np.std(br),
        np.max(br),
    ]

    if mode == "hr_br_rr":
        rr = sequence[:, 2]

        values.extend(
            [
                np.mean(rr),
                np.std(rr),
                np.min(rr),
            ]
        )

    return np.asarray(
        values,
        dtype=float,
    )


def load_dataset_v2(
    mode="hr_br",
):
    if mode not in {
        "hr_br",
        "hr_br_rr",
    }:
        raise ValueError(
            "mode must be hr_br or hr_br_rr"
        )

    sequences = []
    statistics = []
    labels = []
    groups = []
    sports = []
    metadata = []

    used_recordings = set()
    skipped_recordings = []
    ignored_segments = Counter()
    rr_quality = {}

    for data_path in sorted(
        SPORTDB_ROOT.rglob(
            "Data.mat"
        )
    ):
        relative = data_path.relative_to(
            SPORTDB_ROOT
        )

        sport = relative.parts[0]

        if sport not in SPORTS:
            continue

        content = loadmat(
            data_path,
            squeeze_me=True,
            struct_as_record=False,
        )

        data = content[
            "Data"
        ]

        fields = set(
            getattr(
                data,
                "_fieldnames",
                [],
            )
        )

        required = {
            "HR",
            "BR",
        }

        if not required.issubset(
            fields
        ):
            continue

        if (
            mode == "hr_br_rr"
            and "RR" not in fields
        ):
            continue

        hr = np.asarray(
            getattr(
                data,
                "HR",
            ),
            dtype=float,
        ).reshape(-1)

        br = np.asarray(
            getattr(
                data,
                "BR",
            ),
            dtype=float,
        ).reshape(-1)

        if "RR" in fields:
            rr = np.asarray(
                getattr(
                    data,
                    "RR",
                ),
                dtype=float,
            ).reshape(-1)
        else:
            rr = None

        lengths = [
            len(hr),
            len(br),
        ]

        if rr is not None:
            lengths.append(
                len(rr)
            )

        signal_length = min(
            lengths
        )

        if signal_length < WINDOW_SIZE:
            continue

        hr = hr[
            :signal_length
        ]

        br = br[
            :signal_length
        ]

        if rr is not None:
            rr = rr[
                :signal_length
            ]

        segments = parse_annotations(
            data_path.with_name(
                "TrNote.txt"
            )
        )

        if not segments:
            continue

        (
            subject,
            session,
            recording,
        ) = subject_and_session(
            data_path
        )

        mapped = mapped_segments(
            sport,
            segments,
            signal_length,
        )

        mapped_raw = {
            item[
                "segment_index"
            ]
            for item in mapped
        }

        for index, segment in enumerate(
            segments
        ):
            if (
                index not in mapped_raw
                and SPORT_LABELS[
                    sport
                ].get(
                    segment[
                        "normalized_label"
                    ]
                )
                is None
            ):
                ignored_segments[
                    f"{sport}:"
                    f"{segment['raw_label']}"
                ] += 1

        if mode == "hr_br_rr":
            coverage = (
                rr_state_coverage(
                    rr,
                    mapped,
                )
            )

            rr_quality[
                recording
            ] = coverage

            if any(
                coverage[
                    state
                ]
                < RR_MIN_STATE_COVERAGE
                for state
                in LABEL_TO_ID
            ):
                skipped_recordings.append(
                    (
                        recording,
                        {
                            state:
                            round(
                                coverage[
                                    state
                                ],
                                4,
                            )
                            for state
                            in LABEL_TO_ID
                        },
                    )
                )
                continue

        recording_windows = 0

        for segment in mapped:
            state = segment[
                "state"
            ]

            label = LABEL_TO_ID[
                state
            ]

            for start in range(
                segment[
                    "start_index"
                ],
                segment[
                    "end_index"
                ]
                - WINDOW_SIZE
                + 1,
                STEP_SIZE,
            ):
                end = (
                    start
                    + WINDOW_SIZE
                )

                if mode == "hr_br":
                    sequence = (
                        np.column_stack(
                            (
                                hr[
                                    start:end
                                ],
                                br[
                                    start:end
                                ],
                            )
                        )
                    )
                else:
                    sequence = (
                        np.column_stack(
                            (
                                hr[
                                    start:end
                                ],
                                br[
                                    start:end
                                ],
                                rr[
                                    start:end
                                ],
                            )
                        )
                    )

                if not np.isfinite(
                    sequence
                ).all():
                    continue

                sequences.append(
                    sequence
                )

                statistics.append(
                    statistical_features(
                        sequence,
                        mode,
                    )
                )

                labels.append(
                    label
                )

                groups.append(
                    subject
                )

                sports.append(
                    sport
                )

                metadata.append(
                    {
                        "sport":
                        sport,

                        "subject":
                        subject,

                        "session":
                        session,

                        "recording":
                        recording,

                        "state":
                        state,

                        "raw_state":
                        segment[
                            "raw_label"
                        ],

                        "segment_index":
                        segment[
                            "segment_index"
                        ],

                        "segment_start_sec":
                        segment[
                            "original_start"
                        ],

                        "segment_end_sec":
                        segment[
                            "original_end"
                        ],

                        "clipped_segment_start_sec":
                        segment[
                            "clipped_start"
                        ],

                        "clipped_segment_end_sec":
                        segment[
                            "clipped_end"
                        ],

                        "truncated_seconds":
                        segment[
                            "truncated_seconds"
                        ],

                        "window_start_sample":
                        start,

                        "window_end_sample":
                        end,

                        "window_start_sec":
                        float(
                            start
                        ),

                        "window_end_sec":
                        float(
                            end
                        ),

                        "samples_per_second":
                        SIGNAL_RATE_HZ,
                    }
                )

                recording_windows += 1

        if recording_windows:
            used_recordings.add(
                recording
            )

    return (
        np.asarray(
            sequences,
            dtype=float,
        ),
        np.asarray(
            statistics,
            dtype=float,
        ),
        np.asarray(
            labels,
            dtype=int,
        ),
        np.asarray(
            groups,
            dtype=str,
        ),
        np.asarray(
            sports,
            dtype=str,
        ),
        metadata,
        sorted(
            used_recordings
        ),
        skipped_recordings,
        ignored_segments,
        rr_quality,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "hr_br",
            "hr_br_rr",
        ],
        default="hr_br",
    )

    args = parser.parse_args()

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
        args.mode
    )

    print("=" * 80)
    print("DATASET LOADER V2")
    print("=" * 80)

    print(
        f"mode={args.mode}"
    )

    print(
        f"recordings={len(used)}"
    )

    print(
        f"subjects={len(set(groups))}"
    )

    print(
        f"windows={len(y)}"
    )

    print(
        f"sequence_shape={x_seq.shape}"
    )

    print(
        f"statistical_shape={x_stat.shape}"
    )

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

    print(
        "subjects_per_sport="
        + str(
            {
                sport:
                len(
                    {
                        group
                        for group, value
                        in zip(
                            groups,
                            sports,
                        )
                        if value
                        == sport
                    }
                )
                for sport
                in sorted(
                    set(
                        sports
                    )
                )
            }
        )
    )

    truncated = {}

    for item in metadata:
        seconds = item[
            "truncated_seconds"
        ]

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

    print("")

    print(
        "truncated_seconds="
        + str(
            {
                key:
                round(
                    value,
                    1,
                )
                for key, value
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

    print(
        "skipped_recordings="
        + str(
            skipped
        )
    )


if __name__ == "__main__":
    main()