import {
  CheckCircle2,
  Search,
  AlertCircle,
} from "lucide-react";

import type {
  SportSession,
  ModelPrediction,
} from "../data/sessionModels";

import {
  getHitlCounts,
} from "../utils/sessionReview";

interface Props {
  session: SportSession;
  onSelect: (prediction: ModelPrediction) => void;
}

const categories = [
  {
    id: "Accept",
    label: "Acceptées",
    description: "Validation non requise",
    color: "#32946a",
    icon: CheckCircle2,
  },
  {
    id: "Check",
    label: "À vérifier",
    description: "Vérification conseillée",
    color: "#b38932",
    icon: Search,
  },
  {
    id: "Review",
    label: "À examiner",
    description: "Revue humaine recommandée",
    color: "#c34d4d",
    icon: AlertCircle,
  },
] as const;

export default function HitlOverview({
  session,
  onSelect,
}: Props) {
  const counts = getHitlCounts(session);
  const total = session.predictions.length;

  if (!total) return null;

  return (
    <section className="panel hitl-overview">
      <div className="panel-heading">
        <div>
          <h2>Politique de décision RHYVA</h2>
          <p>
            Résultats de l'expérimentation HITL robuste
          </p>
        </div>
      </div>

      <div className="hitl-distribution">
        {categories.map(({ id, color }) => (
          <div
            key={id}
            style={{
              background: color,
              width: `${counts[id] / total * 100}%`,
            }}
            title={`${id} : ${counts[id]}`}
          />
        ))}
      </div>

      <div className="hitl-summary-grid">
        {categories.map(
          ({ id, label, description, color, icon: Icon }) => {
            const windows = session.predictions.filter(
              (p) => p.hitlDecision === id
            );

            return (
              <button
                key={id}
                type="button"
                className="hitl-summary-card"
                disabled={!windows.length}
                onClick={() => {
                  const example =
                    windows[Math.floor(windows.length / 2)];

                  if (example) onSelect(example);
                }}
              >
                <Icon size={23} style={{ color }} />

                <small>{id} · {label}</small>

                <strong style={{ color }}>
                  {counts[id]}
                </strong>

                <span>
                  {(counts[id] / total * 100).toFixed(1)} %
                </span>

                <small>{description}</small>

                <span className="hitl-example-link">
                  {windows.length
                    ? "Explorer un exemple"
                    : "Aucune fenêtre"}
                </span>
              </button>
            );
          }
        )}
      </div>

      <small className="helper-text">
        Ces décisions ne garantissent pas l'exactitude
        des prédictions. Sélectionnez une catégorie
        pour explorer une fenêtre correspondante.
      </small>
    </section>
  );
}
