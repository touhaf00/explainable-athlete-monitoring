from collections import Counter, defaultdict
import os
import re

import numpy as np
import pandas as pd

from deep_learning_multisport import (
    load_multisport_windows,
    LABEL_NAMES,
    SPORTS_CONFIG,
    SPORTDB_ROOT,
)


def annotation_mapping_report():
    pattern = re.compile(
        r"([A-Za-z0-9\s\-_+()]+?)\s*:\s*from\s+"
        r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s+to\s+"
        r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",
        re.IGNORECASE,
    )

    unmapped = Counter()
    ambiguous = Counter()

    for sport, config in SPORTS_CONFIG.items():
        sport_path = os.path.join(
            SPORTDB_ROOT,
            sport,
        )

        if not os.path.isdir(sport_path):
            continue

        for dirpath, _, filenames in os.walk(sport_path):
            if "TrNote.txt" not in filenames:
                continue

            path = os.path.join(
                dirpath,
                "TrNote.txt",
            )

            with open(
                path,
                "r",
                encoding=config["encoding"],
                errors="replace",
            ) as file:
                text = file.read()

            for match in pattern.finditer(text):
                raw = match.group(1).strip()
                normalized = raw.lower()

                states = {
                    state
                    for key, state in config["state_map"].items()
                    if key in normalized
                }

                if not states:
                    unmapped[(sport, raw)] += 1

                elif len(states) > 1:
                    ambiguous[(sport, raw)] += 1

    return unmapped, ambiguous


def main():
    (
        x_seq,
        x_stat,
        y,
        groups,
        sports,
        meta,
        used,
        skipped,
        counts,
    ) = load_multisport_windows()

    print("=" * 70)
    print("MULTI-SPORT DATA QUALITY")
    print("=" * 70)

    print("")
    print("Dataset")

    print(f"files_used={len(used)}")
    print(f"files_skipped={len(skipped)}")
    print(f"groups={len(set(groups))}")
    print(f"windows={len(y)}")

    class_distribution = {
        LABEL_NAMES[k]: int(v)
        for k, v in Counter(y).items()
    }

    print(
        "class_distribution="
        + str(class_distribution)
    )

    print(
        "windows_per_sport="
        + str(dict(counts))
    )

    rate_rows = {}

    for group, sport, item in zip(
        groups,
        sports,
        meta,
    ):
        key = str(group)

        if key not in rate_rows:
            rate_rows[key] = (
                str(sport),
                float(
                    item.get(
                        "samples_per_second",
                        np.nan,
                    )
                ),
            )

    rate_df = pd.DataFrame(
        [
            {
                "group": group,
                "sport": sport,
                "samples_per_second": rate,
            }
            for group, (sport, rate)
            in rate_rows.items()
        ]
    )

    rate_summary = (
        rate_df
        .groupby("sport")["samples_per_second"]
        .agg(
            [
                "min",
                "median",
                "max",
            ]
        )
        .round(4)
        .to_dict("index")
    )

    print("")
    print("Sampling rate")

    print(
        "sampling_rate_by_sport="
        + str(rate_summary)
    )

    feature_names = [
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

    frame = pd.DataFrame(
        x_stat,
        columns=feature_names,
    )

    frame["label"] = y

    contradictory_stats = (
        frame
        .groupby(
            feature_names,
            dropna=False,
        )["label"]
        .nunique()
        .gt(1)
        .sum()
    )

    exact_sequences = defaultdict(set)

    for sequence, label, group in zip(
        x_seq,
        y,
        groups,
    ):
        key = (
            str(group),
            sequence.tobytes(),
        )

        exact_sequences[key].add(
            int(label)
        )

    contradictory_sequences = sum(
        len(labels) > 1
        for labels in exact_sequences.values()
    )

    print("")
    print("Contradictory labels")

    print(
        "contradictory_stat_windows="
        f"{int(contradictory_stats)}"
    )

    print(
        "contradictory_exact_sequences="
        f"{int(contradictory_sequences)}"
    )

    if (
        contradictory_stats
        or contradictory_sequences
    ):
        raise SystemExit(
            "ERROR: contradictory labels remain."
        )

    provenance = pd.DataFrame(
        [
            {
                "group": str(group),
                "sport": str(sport),
                "segment_start_sec": float(
                    item.get(
                        "segment_start_sec",
                        np.nan,
                    )
                ),
                "window_start_sample": int(
                    item.get(
                        "window_start_sample",
                        -1,
                    )
                ),
            }
            for group, sport, item in zip(
                groups,
                sports,
                meta,
            )
        ]
    )

    chronological = True

    for _, part in provenance.groupby(
        "group",
        sort=False,
    ):
        ordered = list(
            zip(
                part["segment_start_sec"],
                part["window_start_sample"],
            )
        )

        if any(
            ordered[i] > ordered[i + 1]
            for i in range(
                len(ordered) - 1
            )
        ):
            chronological = False
            break

    print("")
    print("Chronology")

    print(
        f"chronological_windows="
        f"{chronological}"
    )

    if not chronological:
        raise SystemExit(
            "ERROR: windows are not chronological."
        )

    unmapped, ambiguous = (
        annotation_mapping_report()
    )

    print("")
    print("Annotation mapping")

    print(
        "unmapped_annotations="
        + str(
            {
                f"{sport}:{label}": count
                for (
                    sport,
                    label,
                ), count in unmapped.items()
            }
        )
    )

    print(
        "ambiguous_annotations="
        + str(
            {
                f"{sport}:{label}": count
                for (
                    sport,
                    label,
                ), count in ambiguous.items()
            }
        )
    )

    print("")
    print("=" * 70)
    print("DATA QUALITY CHECK PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
