from collections import Counter, defaultdict
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


BASE_DIR = Path(__file__).resolve().parents[2]
SPORTDB_ROOT = BASE_DIR / "SportDB2"

INTERVAL_PATTERN = re.compile(
    r"([^\r\n:]+?)\s*:\s*from\s+"
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


def normalize_text(value):
    return (
        value
        .replace("\ufeff", "")
        .replace("ï»¿", "")
        .strip()
    )


def load_data(path):
    content = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
    )

    return content["Data"]


def data_fields(data):
    return tuple(
        sorted(
            getattr(
                data,
                "_fieldnames",
                [],
            )
        )
    )


def parse_demographics(path):
    if not path.exists():
        return {}

    text = path.read_text(
        encoding="latin-1",
        errors="replace",
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return {}

    headers = re.split(
        r"\s+",
        lines[0],
    )

    values = re.split(
        r"\s+",
        lines[1],
    )

    if len(headers) != len(values):
        return {}

    return dict(
        zip(
            headers,
            values,
        )
    )


def parse_trnote(path):
    if not path.exists():
        return {
            "type": "missing",
            "intervals": [],
            "lines": [],
        }

    text = path.read_text(
        encoding="latin-1",
        errors="replace",
    )

    matches = list(
        INTERVAL_PATTERN.finditer(
            text
        )
    )

    if matches:
        intervals = []

        for match in matches:
            intervals.append(
                {
                    "label": normalize_text(
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

        return {
            "type": "interval",
            "intervals": intervals,
            "lines": [],
        }

    lines = [
        normalize_text(line)
        for line in text.splitlines()
        if normalize_text(line)
    ]

    return {
        "type": "named",
        "intervals": [],
        "lines": lines,
    }


def main():
    if not SPORTDB_ROOT.exists():
        raise FileNotFoundError(
            SPORTDB_ROOT
        )

    records = []

    field_sets = defaultdict(
        Counter
    )

    annotations = defaultdict(
        Counter
    )

    named_annotations = defaultdict(
        Counter
    )

    demographic_fields = Counter()
    demographic_missing = Counter()
    demographic_total = Counter()

    for data_path in sorted(
        SPORTDB_ROOT.rglob(
            "Data.mat"
        )
    ):
        relative = data_path.relative_to(
            SPORTDB_ROOT
        )

        sport = relative.parts[0]

        data = load_data(
            data_path
        )

        fields = data_fields(
            data
        )

        field_sets[sport][fields] += 1

        field_set = set(
            fields
        )

        direct_hr_rr = (
            "HR" in field_set
            and "RR" in field_set
        )

        split_hr = sorted(
            field
            for field in fields
            if field.startswith(
                "HR_"
            )
        )

        split_rr = sorted(
            field
            for field in fields
            if field.startswith(
                "RR_"
            )
        )

        ecg_fields = sorted(
            field
            for field in fields
            if field.startswith(
                "ECG"
            )
        )

        recovery_fields = sorted(
            field
            for field in fields
            if "recovery"
            in field.lower()
        )

        resting_fields = sorted(
            field
            for field in fields
            if "rest"
            in field.lower()
        )

        trnote = parse_trnote(
            data_path.with_name(
                "TrNote.txt"
            )
        )

        first_start = np.nan
        last_end = np.nan
        annotated_span = np.nan

        if trnote["type"] == "interval":
            for interval in trnote[
                "intervals"
            ]:
                annotations[
                    sport
                ][
                    interval["label"]
                ] += 1

            first_start = min(
                item["start"]
                for item in trnote[
                    "intervals"
                ]
            )

            last_end = max(
                item["end"]
                for item in trnote[
                    "intervals"
                ]
            )

            annotated_span = (
                last_end
                - first_start
            )

        elif trnote["type"] == "named":
            for line in trnote[
                "lines"
            ]:
                named_annotations[
                    sport
                ][line] += 1

        demographics = (
            parse_demographics(
                data_path.with_name(
                    "Dem.txt"
                )
            )
        )

        for key, value in (
            demographics.items()
        ):
            demographic_fields[
                key
            ] += 1

            demographic_total[
                key
            ] += 1

            if value.upper() == "NA":
                demographic_missing[
                    key
                ] += 1

        hr_length = np.nan
        rr_length = np.nan
        br_length = np.nan
        ecg_length = np.nan
        ecg_hr_ratio = np.nan
        hr_per_annotated_second = np.nan

        if "HR" in field_set:
            hr_length = np.asarray(
                getattr(
                    data,
                    "HR",
                )
            ).size

        if "RR" in field_set:
            rr_length = np.asarray(
                getattr(
                    data,
                    "RR",
                )
            ).size

        if "BR" in field_set:
            br_length = np.asarray(
                getattr(
                    data,
                    "BR",
                )
            ).size

        if "ECG" in field_set:
            ecg_length = np.asarray(
                getattr(
                    data,
                    "ECG",
                )
            ).size

        if (
            np.isfinite(
                ecg_length
            )
            and np.isfinite(
                hr_length
            )
            and hr_length > 0
        ):
            ecg_hr_ratio = (
                ecg_length
                / hr_length
            )

        if (
            np.isfinite(
                hr_length
            )
            and np.isfinite(
                annotated_span
            )
            and annotated_span > 0
        ):
            hr_per_annotated_second = (
                hr_length
                / annotated_span
            )

        records.append(
            {
                "sport": sport,
                "path": str(
                    relative.parent
                ),
                "fields": fields,
                "direct_hr_rr":
                direct_hr_rr,
                "split_hr_rr":
                bool(
                    split_hr
                    and split_rr
                ),
                "has_br":
                "BR"
                in field_set,
                "has_ecg":
                bool(
                    ecg_fields
                ),
                "recovery_fields":
                recovery_fields,
                "resting_fields":
                resting_fields,
                "trnote_type":
                trnote["type"],
                "first_start":
                first_start,
                "last_end":
                last_end,
                "annotated_span":
                annotated_span,
                "hr_length":
                hr_length,
                "rr_length":
                rr_length,
                "br_length":
                br_length,
                "ecg_length":
                ecg_length,
                "ecg_hr_ratio":
                ecg_hr_ratio,
                "hr_per_annotated_second":
                hr_per_annotated_second,
            }
        )

    sports = sorted(
        {
            row["sport"]
            for row in records
        }
    )

    print(
        "=" * 80
    )

    print(
        "SPORTDB2 FULL EXPLORATION"
    )

    print(
        "=" * 80
    )

    print(
        f"recordings={len(records)}"
    )

    print(
        f"sports={len(sports)}"
    )

    print(
        f"sport_names={sports}"
    )

    print("")

    print(
        "=" * 80
    )

    print(
        "DATA AVAILABILITY BY SPORT"
    )

    print(
        "=" * 80
    )

    for sport in sports:
        rows = [
            row
            for row in records
            if row["sport"]
            == sport
        ]

        print("")
        print(
            sport
        )

        print(
            f"  recordings={len(rows)}"
        )

        print(
            "  direct_HR_RR="
            f"{sum(row['direct_hr_rr'] for row in rows)}"
        )

        print(
            "  split_HR_RR="
            f"{sum(row['split_hr_rr'] for row in rows)}"
        )

        print(
            "  BR="
            f"{sum(row['has_br'] for row in rows)}"
        )

        print(
            "  ECG="
            f"{sum(row['has_ecg'] for row in rows)}"
        )

        print(
            "  interval_TrNote="
            f"{sum(row['trnote_type'] == 'interval' for row in rows)}"
        )

        print(
            "  named_TrNote="
            f"{sum(row['trnote_type'] == 'named' for row in rows)}"
        )

        print(
            "  recovery_fields="
            f"{sum(bool(row['recovery_fields']) for row in rows)}"
        )

    print("")
    print(
        "=" * 80
    )

    print(
        "FIELD SETS"
    )

    print(
        "=" * 80
    )

    for sport in sports:
        print("")
        print(
            sport
        )

        for (
            fields,
            count,
        ) in field_sets[
            sport
        ].most_common():

            print(
                f"  {count:3d} x "
                + ", ".join(
                    fields
                )
            )

    print("")
    print(
        "=" * 80
    )

    print(
        "INTERVAL ANNOTATIONS"
    )

    print(
        "=" * 80
    )

    for sport in sports:
        if not annotations[
            sport
        ]:
            continue

        print("")
        print(
            sport
        )

        for (
            label,
            count,
        ) in annotations[
            sport
        ].most_common():

            print(
                f"  {count:3d} x "
                f"{label}"
            )

    print("")
    print(
        "=" * 80
    )

    print(
        "NAMED OR EVENT ANNOTATIONS"
    )

    print(
        "=" * 80
    )

    for sport in sports:
        if not named_annotations[
            sport
        ]:
            continue

        print("")
        print(
            sport
        )

        for (
            label,
            count,
        ) in named_annotations[
            sport
        ].most_common():

            print(
                f"  {count:3d} x "
                f"{label}"
            )

    print("")
    print(
        "=" * 80
    )

    print(
        "TIMING AND SAMPLING"
    )

    print(
        "=" * 80
    )

    for sport in sports:
        rows = [
            row
            for row in records
            if (
                row["sport"]
                == sport
                and row[
                    "direct_hr_rr"
                ]
            )
        ]

        if not rows:
            continue

        starts = np.array(
            [
                row[
                    "first_start"
                ]
                for row in rows
                if np.isfinite(
                    row[
                        "first_start"
                    ]
                )
            ]
        )

        hz = np.array(
            [
                row[
                    "hr_per_annotated_second"
                ]
                for row in rows
                if np.isfinite(
                    row[
                        "hr_per_annotated_second"
                    ]
                )
            ]
        )

        ratios = np.array(
            [
                row[
                    "ecg_hr_ratio"
                ]
                for row in rows
                if np.isfinite(
                    row[
                        "ecg_hr_ratio"
                    ]
                )
            ]
        )

        print("")
        print(
            sport
        )

        if starts.size:
            print(
                "  first_annotation_sec="
                f"min:{starts.min():.1f} "
                f"median:{np.median(starts):.1f} "
                f"max:{starts.max():.1f}"
            )

        if hz.size:
            print(
                "  HR_length_over_annotation_span="
                f"min:{hz.min():.4f} "
                f"median:{np.median(hz):.4f} "
                f"max:{hz.max():.4f}"
            )

        if ratios.size:
            print(
                "  ECG_length_over_HR_length="
                f"min:{ratios.min():.1f} "
                f"median:{np.median(ratios):.1f} "
                f"max:{ratios.max():.1f}"
            )

    print("")
    print(
        "=" * 80
    )

    print(
        "SPECIAL RECORDINGS"
    )

    print(
        "=" * 80
    )

    for row in records:
        if (
            row[
                "split_hr_rr"
            ]
            or row[
                "recovery_fields"
            ]
        ):
            print("")
            print(
                row["path"]
            )

            if row[
                "split_hr_rr"
            ]:
                print(
                    "  split_HR_RR=True"
                )

            if row[
                "recovery_fields"
            ]:
                print(
                    "  recovery_fields="
                    + ", ".join(
                        row[
                            "recovery_fields"
                        ]
                    )
                )

    print("")
    print(
        "=" * 80
    )

    print(
        "DEMOGRAPHICS"
    )

    print(
        "=" * 80
    )

    for field in sorted(
        demographic_fields
    ):
        total = demographic_total[
            field
        ]

        missing = demographic_missing[
            field
        ]

        percentage = (
            100.0
            * missing
            / total
            if total
            else 0.0
        )

        print(
            f"{field}: "
            f"available={total} "
            f"missing={missing} "
            f"missing_pct={percentage:.1f}"
        )


if __name__ == "__main__":
    main()