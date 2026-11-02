from collections import Counter
from pathlib import Path
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


def parse_annotations(path):
    text = path.read_text(
        encoding="latin-1",
        errors="replace",
    )

    segments = []

    for match in PATTERN.finditer(text):
        segments.append(
            {
                "label": match.group(1).strip(),
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

    parts = relative.parts

    sport = parts[0]

    subject_parts = parts[1:-2]

    subject = (
        sport
        + "/"
        + "/".join(
            subject_parts
        )
    )

    session = parts[-2]

    return subject, session


def main():
    rows = []

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

        data = content["Data"]

        fields = set(
            getattr(
                data,
                "_fieldnames",
                [],
            )
        )

        if not {
            "HR",
            "RR",
            "BR",
        }.issubset(fields):
            continue

        hr = np.asarray(
            getattr(
                data,
                "HR",
            )
        ).reshape(-1)

        rr = np.asarray(
            getattr(
                data,
                "RR",
            )
        ).reshape(-1)

        br = np.asarray(
            getattr(
                data,
                "BR",
            )
        ).reshape(-1)

        segments = parse_annotations(
            data_path.with_name(
                "TrNote.txt"
            )
        )

        subject, session = (
            subject_and_session(
                data_path
            )
        )

        signal_length = min(
            len(hr),
            len(rr),
            len(br),
        )

        signal_duration = (
            signal_length
            / SIGNAL_RATE_HZ
        )

        annotated_duration = sum(
            max(
                0.0,
                segment[
                    "relative_end"
                ]
                - segment[
                    "relative_start"
                ],
            )
            for segment in segments
        )

        overlap = 0.0

        for previous, current in zip(
            segments,
            segments[1:],
        ):
            overlap += max(
                0.0,
                previous[
                    "relative_end"
                ]
                - current[
                    "relative_start"
                ],
            )

        out_of_bounds = sum(
            segment[
                "relative_start"
            ] < 0
            or segment[
                "relative_end"
            ] > signal_duration + 2
            for segment in segments
        )

        coverage = (
            annotated_duration
            / signal_duration
            if signal_duration > 0
            else np.nan
        )

        rows.append(
            {
                "sport": sport,
                "subject": subject,
                "session": session,
                "n_hr": len(hr),
                "n_rr": len(rr),
                "n_br": len(br),
                "duration": signal_duration,
                "segments": len(
                    segments
                ),
                "coverage": coverage,
                "overlap": overlap,
                "out_of_bounds":
                out_of_bounds,
                "labels": tuple(
                    sorted(
                        {
                            segment[
                                "label"
                            ]
                            for segment
                            in segments
                        }
                    )
                ),
            }
        )

    print("=" * 80)
    print("DATASET MANIFEST V2")
    print("=" * 80)

    print(
        f"recordings={len(rows)}"
    )

    print(
        "subjects="
        f"{len(set(row['subject'] for row in rows))}"
    )

    print("")

    for sport in sorted(
        SPORTS
    ):
        sport_rows = [
            row
            for row in rows
            if row["sport"]
            == sport
        ]

        if not sport_rows:
            continue

        print(
            "=" * 80
        )

        print(
            sport
        )

        print(
            "=" * 80
        )

        print(
            f"recordings={len(sport_rows)}"
        )

        print(
            "subjects="
            f"{len(set(row['subject'] for row in sport_rows))}"
        )

        coverage = np.array(
            [
                row["coverage"]
                for row
                in sport_rows
            ]
        )

        print(
            "annotation_coverage="
            f"min:{coverage.min():.4f} "
            f"median:{np.median(coverage):.4f} "
            f"max:{coverage.max():.4f}"
        )

        print(
            "sessions_with_overlap="
            f"{sum(row['overlap'] > 0 for row in sport_rows)}"
        )

        print(
            "sessions_out_of_bounds="
            f"{sum(row['out_of_bounds'] > 0 for row in sport_rows)}"
        )

        labels = Counter()

        for row in sport_rows:
            for label in row[
                "labels"
            ]:
                labels[label] += 1

        print(
            "labels="
            + str(
                dict(
                    labels
                )
            )
        )

        print("")

    print(
        "=" * 80
    )

    print(
        "PROBLEMATIC SESSIONS"
    )

    print(
        "=" * 80
    )

    problematic = [
        row
        for row in rows
        if (
            row["overlap"] > 0
            or row[
                "out_of_bounds"
            ] > 0
            or row["coverage"] > 1.02
        )
    ]

    if not problematic:
        print(
            "none"
        )

    for row in problematic:
        print(
            row
        )


if __name__ == "__main__":
    main()
