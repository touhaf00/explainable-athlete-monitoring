import { buildCoachReport } from "../utils/coachReport";
import { Brain, Search, History } from "lucide-react";
import { stateLabels, type SessionReviewTarget } from "../utils/sessionReview";

import {
  reasons,
  verdictLabels,
  emptyStudy,
  type Reason,
  type Verdict,
  type Report,
  type StudyState
} from "../utils/adaptiveReview";

interface Props {
  athlete: SessionReviewTarget;
  decision?: "agree" | "check" | "disagree";
  value?: StudyState;
  onChange: (value: StudyState) => void;
}

function analyze(
  athlete: SessionReviewTarget,
  reason: Reason,
  context: string
): string[] {
  const session = athlete.session;

  const report = buildCoachReport(session);

  const findings: string[] = [];

  if (athlete.window) {
    const p = athlete.window;

    findings.push(
      `Passage étudié : de ${(p.startSecond / 60).toFixed(1)} à ${(p.endSecond / 60).toFixed(1)} minutes.`,
      `État estimé : ${stateLabels[p.predictedState]}.`
    );
  }

  switch (reason) {
    case "context":
      findings.push(
        report.effortText,
        "Les conditions de l'entraînement et le ressenti du sportif peuvent compléter cette interprétation."
      );
      break;

    case "sensor":
      findings.push(
        `Mesures cardiaques disponibles : ${(session.heartRateCoverage * 100).toFixed(0)} %.`,
        `Mesures respiratoires disponibles : ${(session.respiratoryRateCoverage * 100).toFixed(0)} %.`,
        "Vérifier le positionnement du capteur, les interruptions éventuelles et les conditions de mesure."
      );
      break;

    case "reference":
      findings.push(
        report.recoveryText,
        "Comparer avec les précédents entraînements du même type lorsque cet historique sera disponible."
      );
      break;

    case "classification":
      findings.push(
        report.effortText,
        report.recoveryText,
        "Vérifier si le passage correspond réellement à une activité, une pause ou une récupération."
      );
      break;

    case "other":
      findings.push(
        report.conclusion,
        report.recommendation
      );
      break;
  }

  if (context.trim()) {
    findings.push(
      `Contexte renseigné par le coach : ${context.trim()}.`
    );
  }

  return findings;
}

export default function AdaptiveReview({
  athlete,
  decision,
  value,
  onChange
}: Props) {
  const state = value ?? emptyStudy();

  const active =
    state.activeIndex === null
      ? undefined
      : state.reports[state.activeIndex];

  const needsAnalysis =
    decision === "check" || decision === "disagree";

  function launchAnalysis() {
    if (!state.reason) return;

    const report: Report = {
      reason: state.reason,
      context: state.context.trim(),
      findings: analyze(
        athlete,
        state.reason,
        state.context
      )
    };

    onChange({
      ...state,
      reports: [...state.reports, report],
      activeIndex: state.reports.length,
      correction: ""
    });
  }

  function finalize(verdict: Verdict) {
    if (
      state.activeIndex === null ||
      !active ||
      active.verdict
    ) {
      return;
    }

    if (
      verdict === "corrected" &&
      !state.correction.trim()
    ) {
      return;
    }

    onChange({
      ...state,
      reports: state.reports.map((report, index) =>
        index === state.activeIndex
          ? {
              ...report,
              verdict,
              correction: state.correction.trim()
            }
          : report
      )
    });
  }

  if (!needsAnalysis && !state.reports.length) {
    return null;
  }

  return (
    <section className="adaptive-panel">
      <div className="adaptive-heading">
        <Brain size={20} />
        <h3>Examen complémentaire</h3>
      </div>

      <small className="adaptive-caption">
        Approfondir la compréhension de la séance
      </small>

      {needsAnalysis && (
        <>
          <p>
            Votre retour permet à Rhyva de cibler
            les vérifications complémentaires.
          </p>

          <label htmlFor="adaptive-reason">
            Pourquoi souhaitez-vous revoir cette observation ?
          </label>

          <select
            id="adaptive-reason"
            value={state.reason}
            onChange={(event) =>
              onChange({
                ...state,
                reason: event.target.value as Reason | "",
                activeIndex: null,
                correction: ""
              })
            }
          >
            <option value="">Choisir un motif</option>

            {Object.entries(reasons).map(
              ([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              )
            )}
          </select>

          <label htmlFor="adaptive-context">
            Informations sur l'entraînement
          </label>

          <textarea
            id="adaptive-context"
            value={state.context}
            maxLength={500}
            rows={3}
            placeholder="Ajoutez un contexte fictif..."
            onChange={(event) =>
              onChange({
                ...state,
                context: event.target.value,
                activeIndex: null,
                correction: ""
              })
            }
          />

          <button
            type="button"
            className="adaptive-button"
            disabled={!state.reason}
            onClick={launchAnalysis}
          >
            <Search size={17} />
            Approfondir cette observation
          </button>

          {active && (
            <div className="adaptive-report">
              <h4>Rapport complémentaire</h4>

              <small>
                Vérifications effectuées sur les
                données de démonstration.
              </small>

              <ul>
                {analyze(athlete, active.reason, active.context).map((finding, index) => (
                  <li key={index}>{finding}</li>
                ))}
              </ul>

              {!active.verdict ? (
                <>
                  <h4>Votre interprétation</h4>

                  <label htmlFor="adaptive-correction">
                    Votre observation ou correction
                  </label>

                  <textarea
                    id="adaptive-correction"
                    rows={3}
                    maxLength={500}
                    value={state.correction}
                    placeholder="Précisez votre interprétation..."
                    onChange={(event) =>
                      onChange({
                        ...state,
                        correction: event.target.value
                      })
                    }
                  />

                  <button
                    type="button"
                    className="adaptive-choice"
                    onClick={() => finalize("confirmed")}
                  >
                    Maintenir l'observation initiale
                  </button>

                  <button
                    type="button"
                    className="adaptive-choice"
                    disabled={!state.correction.trim()}
                    onClick={() => finalize("corrected")}
                  >
                    Proposer une correction
                  </button>

                  <button
                    type="button"
                    className="adaptive-choice"
                    onClick={() => finalize("insufficient")}
                  >
                    Données insuffisantes
                  </button>
                </>
              ) : (
                <div className="adaptive-result" role="status">
                  <strong>
                    {verdictLabels[active.verdict]}
                  </strong>

                  <p>
                    Votre interprétation a été enregistrée.
                  </p>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {state.reports.length > 0 && (
        <details className="adaptive-history">
          <summary>
            <History size={16} />
            Historique ({state.reports.length})
          </summary>

          {[...state.reports].reverse().map(
            (report, index) => (
              <div className="adaptive-event" key={index}>
                <strong>{reasons[report.reason]}</strong>

                <small>
                  {report.verdict
                    ? verdictLabels[report.verdict]
                    : "En attente de validation"}
                </small>

                {needsAnalysis && !report.verdict && (
                  <button
                    type="button"
                    className="adaptive-choice"
                    onClick={() =>
                      onChange({
                        ...state,
                        reason: report.reason,
                        context: report.context,
                        correction: "",
                        activeIndex:
                          state.reports.length - 1 - index
                      })
                    }
                  >
                    Reprendre cette analyse
                  </button>
                )}

                {report.context && <p>{report.context}</p>}

                {report.correction && (
                  <p>Correction : {report.correction}</p>
                )}
              </div>
            )
          )}
        </details>
      )}

      <small className="adaptive-caption">
        Les observations de test restent dans votre navigateur.
      </small>
    </section>
  );
}
