import { HeartPulse, Wind, Activity } from "lucide-react";

import type {
  ModelPrediction,
  SportSession,
} from "../data/sessionModels";

import {
  getPredictionAtSample,
  stateLabels,
  type SessionReviewTarget,
} from "../utils/sessionReview";

interface Props {
  athlete: SessionReviewTarget;
  activeIndex: number;
  predictionOverride?: ModelPrediction;
}

type Metric = "heartRate" | "respiratoryRate";

function mean(values: number[]) {
  return values.reduce((sum, v) => sum + v, 0)
    / values.length;
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);

  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function number(value: number) {
  return value.toLocaleString("fr-FR", {
    maximumFractionDigits: 1,
  });
}

function time(seconds: number) {
  return `${Math.floor(seconds / 60)} min ${
    String(Math.floor(seconds % 60)).padStart(2, "0")
  } s`;
}

function measurements(
  session: SportSession,
  metric: Metric,
  start: number,
  end: number,
): number[] {
  if (session.aggregationSeconds !== 1) return [];

  const samples = session.samples.filter((sample) => {
    const second = Math.round(sample.minute * 60);
    return second >= start && second < end;
  });

  if (
    samples.length !== end - start ||
    samples.some(
      (sample, index) =>
        Math.round(sample.minute * 60) !== start + index
    )
  ) {
    return [];
  }

  const values = samples.map((sample) => sample[metric]);

  if (
    values.some(
      (value) =>
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        value <= 0
    )
  ) {
    return [];
  }

  return values as number[];
}

function restingReference(
  session: SportSession,
  metric: Metric,
): number | null {
  if (
    !session.initialRestConfirmed ||
    (session.initialRestSeconds ?? 0) < 275
  ) {
    return null;
  }

  const averages: number[] = [];

  for (let start = 0; start <= 225; start += 25) {
    const values = measurements(
      session, metric, start, start + 50
    );

    if (values.length !== 50) return null;

    averages.push(mean(values));
  }

  return median(averages);
}

function describe(
  feature: string,
  values: number[],
  reference: number | null,
  metric: Metric,
) {
  if (!values.length) {
    return "Les mesures détaillées de ce passage ne sont pas disponibles.";
  }

  const unit = metric === "heartRate"
    ? "bpm"
    : "respirations/min";

  const average = mean(values);

  if (
    feature.endsWith("_delta_cal") ||
    feature.endsWith("_ratio_cal") ||
    feature.startsWith("mean_")
  ) {
    return reference !== null
      ? `Fréquence moyenne de ${number(average)} ${unit}, contre ${number(reference)} ${unit} au repos initial.`
      : `Fréquence moyenne de ${number(average)} ${unit} sur ce passage.`;
  }

  if (feature.startsWith("max_")) {
    return `Fréquence maximale de ${number(Math.max(...values))} ${unit} sur ce passage.`;
  }

  if (feature.startsWith("std_")) {
    return `Les mesures fluctuent entre ${number(Math.min(...values))} et ${number(Math.max(...values))} ${unit}.`;
  }

  if (feature.startsWith("slope_")) {
    const center = (values.length - 1) / 2;

    const numerator = values.reduce(
      (sum, value, index) =>
        sum + (index - center) * (value - average),
      0
    );

    const denominator = values.reduce(
      (sum, _, index) =>
        sum + (index - center) ** 2,
      0
    );

    const change = denominator
      ? numerator / denominator * (values.length - 1)
      : 0;

    return Math.abs(change) < 0.1
      ? "La fréquence reste globalement stable sur ce passage."
      : `Tendance ${change > 0 ? "à la hausse" : "à la baisse"} sur ce passage, d'environ ${number(Math.abs(change))} ${unit}.`;
  }

  if (feature.startsWith("delta_")) {
    const size = Math.max(
      1, Math.min(10, Math.floor(values.length / 5))
    );

    const start = mean(values.slice(0, size));
    const end = mean(values.slice(-size));

    return `La fréquence moyenne passe de ${number(start)} à ${number(end)} ${unit} entre le début et la fin du passage.`;
  }

  return `Fréquence moyenne observée : ${number(average)} ${unit}.`;
}

export default function ExplainabilityPanel({
  athlete,
  activeIndex,
  predictionOverride,
}: Props) {
  const session = athlete.session;

  const prediction =
    predictionOverride ??
    getPredictionAtSample(session, activeIndex);

  if (!prediction) return null;

  const factors = [...(prediction.explanationFactors ?? [])]
    .filter(
      (factor) =>
        Number.isFinite(factor.contribution) &&
        factor.contribution > 0
    )
    .sort((a, b) => b.contribution - a.contribution);

  const observations = (["HR", "BR"] as const)
    .flatMap((concept) => {
      const factor = factors.find(
        (item) => item.concept === concept
      );

      if (!factor) return [];

      const metric: Metric = concept === "HR"
        ? "heartRate"
        : "respiratoryRate";

      const values = measurements(
        session,
        metric,
        prediction.startSecond,
        prediction.endSecond,
      );

      return [{
        concept,
        title: concept === "HR"
          ? "Réponse cardiaque"
          : "Réponse respiratoire",
        text: describe(
          factor.feature,
          values,
          restingReference(session, metric),
          metric,
        ),
        contribution: factor.contribution,
      }];
    })
    .sort((a, b) => b.contribution - a.contribution);

  const afterEffort =
    session.effortEndSecond !== undefined &&
    prediction.startSecond >= session.effortEndSecond;

  const unexpectedEffort =
    afterEffort && prediction.predictedState === "effort";

  const needsAttention =
    prediction.hitlDecision !== "Accept" ||
    unexpectedEffort;

  return (
    <section className="rv-xai-panel">
      <div className="rv-xai-heading">
        <Activity size={22} />

        <div>
          <h3>Comprendre ce passage</h3>
          <p>
            {time(prediction.startSecond)}
            {" – "}
            {time(prediction.endSecond)}
          </p>
        </div>
      </div>

      <div className="rv-xai-metric">
        <span>État estimé du sportif</span>

        <strong>
          {stateLabels[prediction.predictedState]}
        </strong>

        {afterEffort && (
          <span>
            Passage situé après la fin déclarée de l'effort.
          </span>
        )}
      </div>

      <h4>Ce qui explique cette interprétation</h4>

      {observations.length > 0 ? (
        <div className="rv-sport-observations">
          {observations.map((item) => {
            const Icon = item.concept === "HR"
              ? HeartPulse
              : Wind;

            return (
              <article
                className="rv-sport-observation"
                key={item.concept}
              >
                <Icon
                  size={24}
                  color={
                    item.concept === "HR"
                      ? "#df8975"
                      : "#5089bf"
                  }
                />

                <div>
                  <strong>{item.title}</strong>
                  <p>{item.text}</p>
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <p className="rv-muted">
          Aucune explication détaillée disponible
          pour ce passage.
        </p>
      )}

      {needsAttention && (
        <div className="rv-xai-notice">
          <div>
            <strong>Point d'attention</strong>

            <p>
              {unexpectedEffort
                ? "Vérifier si l'effort était réellement terminé et recueillir le ressenti du sportif."
                : "Comparer ces mesures avec l'intensité prévue, les conditions de séance et le ressenti du sportif."}
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
