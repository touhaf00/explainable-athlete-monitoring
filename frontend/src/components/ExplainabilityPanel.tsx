import type {
  ModelPrediction,
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

export default function ExplainabilityPanel({
  athlete,
  activeIndex,
  predictionOverride,
}: Props) {
  const prediction =
    predictionOverride ??
    getPredictionAtSample(
      athlete.session,
      activeIndex
    );

  if (!prediction) {
    return <p>Aucune estimation pour cet instant.</p>;
  }

  return (
    <div>
      <h3>État physiologique estimé</h3>

      <strong>
        {stateLabels[prediction.predictedState]}
      </strong>

      <p>
        {prediction.hitlDecision === "Accept"
          ? "Aucune vérification demandée pour cet instant."
          : "Un complément de contexte peut être utile."}
      </p>
    </div>
  );
}
