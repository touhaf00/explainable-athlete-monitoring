from collections import Counter

import numpy as np
import pandas as pd

from src.core.dataset import (
    statistical_features,
    WINDOW_SIZE,
    STEP_SIZE,
)

from src.core.modeling import (
    slope,
    edge_delta,
    feature_sets,
    CALIBRATION_WINDOWS,
)

from src.core.calibration import (
    apply_calibrator,
    predicted_confidence,
)

from src.core.semantic import (
    shap_matrix,
    semantic_consistency,
)

from src.api.policy import (
    load_artifacts,
    get_policy,
    decide,
)


STATES = {
    0: "rest",
    1: "effort",
    2: "recovery",
}

SPORTS = {
    "RUN",
    "CYC",
    "AMF",
    "SOC",
}

REFERENCE_SECONDS = (
    WINDOW_SIZE
    + (CALIBRATION_WINDOWS - 1) * STEP_SIZE
)


def json_number(value):
    value = float(value)

    return value if np.isfinite(value) else None


def raw_features(sequence):
    names = feature_sets("hr_br")["base_6"]

    stats = statistical_features(
        sequence,
        "hr_br",
    )

    row = dict(
        zip(names, map(float, stats))
    )

    hr = sequence[:, 0]
    br = sequence[:, 1]

    row.update({
        "slope_HR_per_sec": slope(hr),
        "slope_BR_per_sec": slope(br),
        "delta_HR": edge_delta(hr),
        "delta_BR": edge_delta(br),
    })

    return row


def valid_window(sequence):
    return (
        len(sequence) == WINDOW_SIZE
        and np.isfinite(sequence).all()
        and (sequence > 0).all()
    )


def build_phases(predictions, length):
    labels = [None] * length
    sources = [None] * length
    distances = np.full(length, np.inf)

    # Reference de repos explicitement declaree.
    for second in range(REFERENCE_SECONDS):
        labels[second] = "rest"
        sources[second] = "declared-rest"

    # Resolution des fenetres qui se chevauchent.
    # On retient la prediction dont le centre
    # est le plus proche de l'instant.
    for p in predictions:
        start = p["startSecond"]
        end = p["endSecond"]
        center = (start + end) / 2

        for second in range(start, end):
            distance = abs(second + 0.5 - center)

            if distance < distances[second]:
                distances[second] = distance
                labels[second] = p["predictedState"]
                sources[second] = "model"

    phases = []
    start = 0

    for second in range(1, length + 1):
        boundary = (
            second == length
            or labels[second] != labels[start]
            or sources[second] != sources[start]
        )

        if boundary:
            if labels[start] is not None:
                phases.append({
                    "state": labels[start],
                    "startMinute": start / 60,
                    "endMinute": second / 60,
                    "source": sources[start],
                })

            start = second

    coverage = sum(
        label is not None
        for label in labels[REFERENCE_SECONDS:]
    ) / max(1, length - REFERENCE_SECONDS)

    return phases, coverage


def analyze_session(
    seconds,
    heart_rate,
    respiratory_rate,
    sport,
    initial_rest_confirmed=False,
    initial_rest_seconds=0,
):
    times = np.asarray(seconds, dtype=float)

    hr = np.asarray(
        heart_rate,
        dtype=float,
    )

    br = np.asarray(
        respiratory_rate,
        dtype=float,
    )

    if not (
        times.ndim == hr.ndim == br.ndim == 1
        and len(times) == len(hr) == len(br)
    ):
        raise ValueError(
            "Les mesures ne sont pas alignees."
        )

    if not 2 <= len(times) <= 10000:
        raise ValueError(
            "Nombre de mesures invalide."
        )

    if (
        not np.isfinite(times).all()
        or (times < 0).any()
    ):
        raise ValueError(
            "Horodatage invalide."
        )

    if not np.allclose(
        np.diff(times),
        1.0,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError(
            "Le modele actuel attend une mesure "
            "par seconde."
        )

    if (
        np.isinf(hr).any()
        or np.isinf(br).any()
        or (hr < 0).any()
        or (br < 0).any()
    ):
        raise ValueError(
            "Mesures physiologiques invalides."
        )

    hr[hr == 0] = np.nan
    br[br == 0] = np.nan

    sequence = np.column_stack([hr, br])

    result = {
        "status": "incomplete",
        "message": "",
        "predictions": [],
        "phases": [],
        "counts": {
            "Accept": 0,
            "Check": 0,
            "Review": 0,
        },
        "analysisCoverage": 0.0,
        "sampleCount": len(times),
    }

    if (
        not np.isfinite(hr).any()
        or not np.isfinite(br).any()
    ):
        result["message"] = (
            "La frequence cardiaque et la respiration "
            "sont necessaires pour cette analyse."
        )
        return result

    if (
        not initial_rest_confirmed
        or initial_rest_seconds < REFERENCE_SECONDS
    ):
        result["message"] = (
            "Une periode initiale de repos confirmee "
            "d'au moins 4 min 35 s est necessaire."
        )
        return result

    if (
        len(sequence)
        < REFERENCE_SECONDS + WINDOW_SIZE
        or initial_rest_seconds > len(sequence)
    ):
        result["message"] = (
            "La seance est trop courte pour "
            "une analyse personnalisee."
        )
        return result

    calibration = []

    for start in range(
        0,
        REFERENCE_SECONDS - WINDOW_SIZE + 1,
        STEP_SIZE,
    ):
        window = sequence[
            start:start + WINDOW_SIZE
        ]

        if not valid_window(window):
            result["message"] = (
                "Les mesures de repos sont incompletes. "
                "Une nouvelle reference est necessaire."
            )
            return result

        calibration.append(
            raw_features(window)
        )

    assert len(calibration) == CALIBRATION_WINDOWS

    cal_hr = float(np.median([
        row["mean_HR"] for row in calibration
    ]))

    cal_br = float(np.median([
        row["mean_BR"] for row in calibration
    ]))

    rows = []
    starts = []

    for start in range(
        REFERENCE_SECONDS,
        len(sequence) - WINDOW_SIZE + 1,
        STEP_SIZE,
    ):
        window = sequence[
            start:start + WINDOW_SIZE
        ]

        if not valid_window(window):
            continue

        row = raw_features(window)

        row.update({
            "mean_HR_delta_cal":
                row["mean_HR"] - cal_hr,

            "mean_BR_delta_cal":
                row["mean_BR"] - cal_br,

            "mean_HR_ratio_cal":
                row["mean_HR"] /
                (abs(cal_hr) + 1e-6),

            "mean_BR_ratio_cal":
                row["mean_BR"] /
                (abs(cal_br) + 1e-6),
        })

        rows.append(row)
        starts.append(start)

    if not rows:
        result["message"] = (
            "Aucune periode exploitable apres le repos."
        )
        return result

    metadata, model, calibrators, profiles = (
        load_artifacts()
    )

    features = metadata["features"]

    assert features == feature_sets(
        "hr_br"
    )["personalized_14"]

    frame = pd.DataFrame(rows)

    x = frame[features].to_numpy(dtype=float)

    predicted, confidence = predicted_confidence(
        model,
        x,
    )

    explanations = shap_matrix(model, x)

    policy = get_policy(sport)

    predictions = []

    for index, state in enumerate(predicted):
        state = int(state)

        calibrated = float(
            apply_calibrator(
                calibrators["by_state"][state],
                [confidence[index]],
            )[0]
        )

        class_position = list(
            model.classes_
        ).index(state)

        consistency, details = semantic_consistency(
            frame.iloc[index],
            state,
            explanations[index, :, class_position],
            features,
            sport,
            profiles,
        )

        reliability = np.sqrt(
            calibrated * consistency
        )

        decision = decide(
            calibrated,
            policy,
        )

        # Discipline hors perimetre :
        # pas d'acceptation automatique.
        if sport not in SPORTS and decision == "Accept":
            decision = "Check"

        start = starts[index]

        predictions.append({
            "startSecond": start,
            "endSecond": start + WINDOW_SIZE,
            "predictedState": STATES[state],

            # Pour un nouveau sportif, la verite
            # terrain n'est pas encore connue.
            "annotatedState": None,
            "correct": None,

            "confidence": float(confidence[index]),
            "calibratedConfidence": calibrated,

            "calibratedReliability":
                json_number(reliability),

            "hitlDecision": decision,

            "acceptThreshold":
                json_number(policy["accept"]),

            "checkThreshold":
                json_number(policy["check"]),

            "thresholdSource": policy["source"],

            "explanationFactors": [
                {
                    "feature": str(feature),
                    "concept": str(concept),
                    "contribution": float(contribution),
                    "similarity": float(similarity),
                }
                for feature, concept,
                    contribution, similarity in details
            ],
        })

    phases, coverage = build_phases(
        predictions,
        len(sequence),
    )

    counts = Counter(
        p["hitlDecision"]
        for p in predictions
    )

    result.update({
        "status":
            "complete" if coverage == 1
            else "partial",

        "message":
            "Analyse experimentale disponible.",

        "modelVersion":
            metadata["baseline_tag"],

        "predictionSource":
            "live-model",

        "calibration": {
            "heartRate": cal_hr,
            "respiratoryRate": cal_br,
            "referenceSeconds": REFERENCE_SECONDS,
        },

        "predictions": predictions,
        "phases": phases,
        "analysisCoverage": coverage,

        "counts": {
            key: counts.get(key, 0)
            for key in [
                "Accept",
                "Check",
                "Review",
            ]
        },
    })

    return result
