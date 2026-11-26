import {
  Brain,
  Search,
  CheckCircle2,
  Clock,
  FilePen,
  Database,
  RotateCcw
} from "lucide-react";

import type { SessionReviewTarget } from "../utils/sessionReview";

import {
  reasons,
  verdictLabels,
  type StudyState
} from "../utils/adaptiveReview";

interface Props {
  athletes: SessionReviewTarget[];

  studies: Partial<Record<string, StudyState>>;

  decisions: Partial<
    Record<string, "agree" | "check" | "disagree">
  >;

  onOpenAthlete: (athlete: SessionReviewTarget) => void;

  onReset: () => void;
}

export default function LearningCenter({
  athletes,
  studies,
  decisions,
  onOpenAthlete,
  onReset
}: Props) {

  const reports = athletes.flatMap((athlete) =>
    (studies[athlete.id]?.reports ?? []).map(
      (report, index) => ({
        athlete,
        report,
        id: `${athlete.id}-${index}`
      })
    )
  );

  const directConfirmations = athletes.filter(
    (athlete) => decisions[athlete.id] === "agree"
  );

  const confirmed = reports.filter(
    ({ report }) => report.verdict === "confirmed"
  );

  const corrected = reports.filter(
    ({ report }) => report.verdict === "corrected"
  );

  const insufficient = reports.filter(
    ({ report }) => report.verdict === "insufficient"
  );

  const pending = reports.filter(
    ({ report }) => !report.verdict
  );

  const candidates =
    confirmed.length +
    corrected.length +
    directConfirmations.length;

  return (
    <>
      <section className="page-heading">
        <span className="eyebrow">
          HUMAN-IN-THE-LOOP
        </span>

        <h1>Centre d'apprentissage</h1>

        <p>
          Vos retours contribuent à préparer
          les futures améliorations de Rhyva.
        </p>
      </section>

      <div className="learning-grid">

        <div className="metric-card">
          <Search size={22} />
          <small>Analyses complémentaires</small>
          <strong>{reports.length}</strong>
          <span>Rapports générés</span>
        </div>

        <div className="metric-card">
          <CheckCircle2 size={22} />
          <small>Confirmations</small>
          <strong>
            {confirmed.length + directConfirmations.length}
          </strong>
          <span>Retours positifs</span>
        </div>

        <div className="metric-card">
          <FilePen size={22} />
          <small>Corrections proposées</small>
          <strong>{corrected.length}</strong>
          <span>À vérifier avant apprentissage</span>
        </div>

        <div className="metric-card">
          <Clock size={22} />
          <small>Rapports non validés</small>
          <strong>{pending.length}</strong>
          <span>En attente de validation</span>
        </div>

      </div>

      <section className="panel learning-panel">

        <div className="panel-heading">
          <div>
            <h2>Préparation à l'apprentissage</h2>
            <p>Exploitation future des retours humains</p>
          </div>

          <Database size={22} className="accent-icon" />
        </div>

        <div className="learning-queue">
          <div>
            <small>Retours candidats à une revue</small>
            <strong>{candidates}</strong>
          </div>

          <div>
            <small>Cas avec données insuffisantes</small>
            <strong>{insufficient.length}</strong>
          </div>

          <div>
            <small>Réentraînements réels</small>
            <strong>0</strong>
          </div>
        </div>

        <p className="learning-notice">
          Les confirmations et corrections sont des
          retours candidats, pas des exemples
          d'apprentissage automatiquement validés.
          Ils doivent être vérifiés et dédupliqués
          avant toute utilisation scientifique.
        </p>

        <div className="learning-pipeline">

          {[
            "Collecte des retours du coach",
            "Vérification des annotations",
            "Préparation des données",
            "Réentraînement contrôlé",
            "Évaluation du nouveau modèle",
            "Déploiement après validation"
          ].map((step, index) => (
            <div className="learning-step" key={step}>

              <span
                className={
                  "learning-step-number" +
                  (index === 0 ? " available" : "")
                }
              >
                {index + 1}
              </span>

              <div>
                <strong>{step}</strong>

                <small>
                  {index === 0
                    ? "Disponible dans la maquette"
                    : "Future intégration scientifique"}
                </small>
              </div>

            </div>
          ))}

        </div>

      </section>

      <section className="panel learning-panel">

        <div className="panel-heading">
          <div>
            <h2>Historique des analyses</h2>
            <p>Retours et corrections des entraîneurs</p>
          </div>

          <Brain size={22} className="accent-icon" />
        </div>

        {reports.length === 0 &&
          directConfirmations.length === 0 && (
            <div className="learning-empty">
              <Brain size={32} />

              <strong>Aucun retour enregistré</strong>

              <p>
                Consultez un sportif et interagissez
                avec son observation pour commencer.
              </p>
            </div>
          )}

        {directConfirmations.map((athlete) => (
          <article
            className="learning-event"
            key={`${athlete.id}-direct`}
          >
            <div className="learning-event-header">
              <strong>
                  {athlete.name}
                  {athlete.window &&
                    ` · ${(athlete.window.startSecond / 60).toFixed(2)} min`}
                </strong>

              <span className="learning-tag confirmed">
                Observation confirmée
              </span>
            </div>

            <p>{athlete.observation}</p>

            <button
              type="button"
              className="learning-link"
              onClick={() => onOpenAthlete(athlete)}
            >
              Consulter la séance
            </button>
          </article>
        ))}

        {[...reports].reverse().map(
          ({ athlete, report, id }) => (
            <article
              className="learning-event"
              key={id}
            >
              <div className="learning-event-header">

                <strong>
                  {athlete.name}
                  {athlete.window &&
                    ` · ${(athlete.window.startSecond / 60).toFixed(2)} min`}
                </strong>

                <span
                  className={
                    "learning-tag " +
                    (report.verdict ?? "pending")
                  }
                >
                  {report.verdict
                    ? verdictLabels[report.verdict]
                    : "En attente de validation"}
                </span>

              </div>

              <p>{reasons[report.reason]}</p>

              {report.context && (
                <p>
                  <strong>Contexte : </strong>
                  {report.context}
                </p>
              )}

              {report.correction && (
                <div className="learning-correction">
                  <strong>Annotation du coach</strong>
                  <p>{report.correction}</p>
                </div>
              )}

              <details>
                <summary>Consulter les vérifications</summary>

                <ul>
                  {report.findings.map(
                    (finding, index) => (
                      <li key={index}>{finding}</li>
                    )
                  )}
                </ul>
              </details>

              <button
                type="button"
                className="learning-link"
                onClick={() => onOpenAthlete(athlete)}
              >
                Consulter la séance
              </button>
            </article>
          )
        )}

      </section>

      <section className="panel learning-panel">
        <h2>Gestion de la démonstration</h2>

        <p className="learning-notice">
          Les retours fictifs sont conservés localement
          si le navigateur autorise le stockage.
          Aucune donnée n'est envoyée à un serveur.
          N'utilisez pas de données personnelles réelles.
        </p>

        <button
          type="button"
          className="learning-reset"
          onClick={onReset}
        >
          <RotateCcw size={17} />
          Réinitialiser les retours du coach
        </button>
      </section>
    </>
  );
}
