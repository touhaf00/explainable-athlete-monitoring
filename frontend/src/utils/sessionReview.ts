import type {
  SportAthlete,
  SportSession,
  ModelPrediction,
} from "../data/sessionModels";

export interface SessionReviewTarget {
  id: string;
  athleteId: string;
  name: string;
  sport: string;
  session: SportSession;
  observation: string;
  status: "normal" | "review" | "check" | "new";
  window?: ModelPrediction;
  statusLabel: string;
}

export function buildReviewTarget(
  athlete: SportAthlete,
  session: SportSession
): SessionReviewTarget {
  const total = session.predictions.length;

  const mismatches = session.predictions.filter(
    (prediction) =>
      prediction.annotatedState !== prediction.predictedState
  ).length;

  return {
    id: session.id,
    athleteId: athlete.id,
    name: athlete.name,
    sport: athlete.sport,
    session,

    observation: total === 0
      ? "Aucune classification IA disponible."
      : `${mismatches} fenêtre(s) sur ${total} présentent une différence entre la prédiction et l'annotation SportDB2.`,

    status: total === 0
      ? "new"
      : mismatches > 0
        ? "review"
        : "normal",

    statusLabel: total === 0
      ? "Analyse indisponible"
      : mismatches > 0
        ? "Classifications à examiner"
        : "Concordance des annotations",
  };
}

export function formatDuration(minutes: number) {
  const seconds = Math.round(minutes * 60);
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  return [
    hours ? `${hours} h` : "",
    `${mins} min`,
    `${secs} s`,
  ].filter(Boolean).join(" ");
}

export const stateLabels = {
  rest: "Repos",
  effort: "Effort",
  recovery: "Récupération",
};


export function getHitlCounts(session: SportSession) {
  const counts = {
    Accept: 0,
    Check: 0,
    Review: 0,
  };

  session.predictions.forEach((p) => {
    counts[p.hitlDecision]++;
  });

  return counts;
}

export function getPredictionAtSample(
  session: SportSession,
  index: number
) {
  const sample = session.samples[index];

  if (!sample) return undefined;

  const second = sample.minute * 60;

  return session.predictions
    .filter(
      (p) =>
        p.startSecond <= second &&
        second < p.endSecond
    )
    .sort(
      (a, b) =>
        Math.abs(
          (a.startSecond + a.endSecond) / 2 - second
        ) -
        Math.abs(
          (b.startSecond + b.endSecond) / 2 - second
        )
    )[0];
}

export function buildWindowReviewTarget(
  target: SessionReviewTarget,
  prediction: ModelPrediction
): SessionReviewTarget {
  return {
    ...target,

    id: `${target.session.id}:${prediction.startSecond}-${prediction.endSecond}`,

    window: prediction,

    observation:
      `Classification proposée : ${stateLabels[prediction.predictedState]}. ` +
      `Fenêtre de ${(prediction.startSecond / 60).toFixed(2)} à ${(prediction.endSecond / 60).toFixed(2)} min.`,

    status:
      prediction.hitlDecision === "Accept"
        ? "normal"
        : prediction.hitlDecision === "Check"
          ? "check"
          : "review",

    statusLabel:
      prediction.hitlDecision === "Accept"
        ? "Accept — Validation non requise"
        : prediction.hitlDecision === "Check"
          ? "Check — Vérification conseillée"
          : "Review — Revue humaine recommandée",
  };
}

export function getWindowSampleIndex(
  session: SportSession,
  prediction: ModelPrediction
) {
  const center =
    (prediction.startSecond + prediction.endSecond) / 120;

  let closest = 0;

  session.samples.forEach((sample, index) => {
    if (
      Math.abs(sample.minute - center) <
      Math.abs(session.samples[closest].minute - center)
    ) {
      closest = index;
    }
  });

  return closest;
}
