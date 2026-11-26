import {
  Activity,
  HeartPulse,
  ShieldCheck,
  BatteryMedium,
  CheckCircle2,
  AlertCircle,
  Eye,
  Info,
} from "lucide-react";

import type {
  SportSession,
} from "../data/sessionModels";

import {
  buildCoachReport,
  type CoachFollowup,
  type CoachAnswer,
} from "../utils/coachReport";

interface Props {
  session: SportSession;
  followup: CoachFollowup;
  onChange: (value: CoachFollowup) => void;
}

const answers = [
  ["unknown", "Non renseigné"],
  ["no", "Non"],
  ["yes", "Oui"],
] as const;

export default function CoachInsight({
  session,
  followup,
  onChange,
}: Props) {
  const report = buildCoachReport(session, followup);

  const StatusIcon =
    report.level === "accept"
      ? CheckCircle2
      : report.level === "check"
        ? Eye
        : report.level === "review"
          ? AlertCircle
          : Info;

  const sections = [
    {
      title: "Réponse à l'effort",
      icon: Activity,
      content: report.effortText,
    },
    {
      title: "Récupération",
      icon: HeartPulse,
      content: report.recoveryText,
    },
    {
      title: "Fatigue et ressenti",
      icon: BatteryMedium,
      content: report.fatigueText,
    },
    {
      title: "Prévention des blessures",
      icon: ShieldCheck,
      content: report.injuryText,
    },
  ];

  function changeAnswer(
    key: "pain" | "unusualFatigue",
    value: CoachAnswer
  ) {
    onChange({
      ...followup,
      [key]: value,
      reviewed: false,
    });
  }

  return (
    <section className="rv-panel rv-final-report">
      <div className="rv-section-heading">
        <div>
          <small className="rv-report-eyebrow">
            SYNTHÈSE DE L'ENTRAÎNEMENT
          </small>

          <h2>Bilan de la séance</h2>

          <p>
            L'essentiel à retenir pour votre suivi sportif.
          </p>
        </div>
      </div>

      <div className={
        `rv-callout rv-${report.level}`
      }>
        <StatusIcon size={28} />

        <div>
          <strong>{report.title}</strong>

          <p>{report.conclusion}</p>

          {followup.reviewed && (
            <small>
              Bilan déjà consulté par le coach.
              Le signal initial reste conservé.
            </small>
          )}
        </div>
      </div>

      <div className="rv-report-sections">
        {sections.map(
          ({ title, icon: Icon, content }) => (
            <article
              className="rv-report-section"
              key={title}
            >
              <div className="rv-report-icon">
                <Icon size={22} />
              </div>

              <div>
                <h3>{title}</h3>
                <p>{content}</p>
              </div>
            </article>
          )
        )}
      </div>

      <div className="rv-recommendation">
        <h3>Conseil pour le suivi</h3>
        <p>{report.recommendation}</p>
      </div>

      <details className="rv-athlete-feedback">
        <summary>
          Compléter le ressenti du sportif
        </summary>

        <p className="rv-muted">
          Ces informations peuvent être recueillies
          après l'entraînement.
        </p>

        <div className="rv-feedback-fields">
          <label>
            Douleur ou gêne inhabituelle

            <select
              value={followup.pain}
              onChange={(event) =>
                changeAnswer(
                  "pain",
                  event.target.value as CoachAnswer
                )
              }
            >
              {answers.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Fatigue inhabituelle

            <select
              value={followup.unusualFatigue}
              onChange={(event) =>
                changeAnswer(
                  "unusualFatigue",
                  event.target.value as CoachAnswer
                )
              }
            >
              {answers.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Difficulté ressentie de la séance

            <select
              value={followup.perceivedEffort ?? ""}
              onChange={(event) =>
                onChange({
                  ...followup,
                  perceivedEffort:
                    event.target.value === ""
                      ? null
                      : Number(event.target.value),
                  reviewed: false,
                })
              }
            >
              <option value="">
                Non renseignée
              </option>

              {Array.from(
                { length: 11 },
                (_, index) => (
                  <option key={index} value={index}>
                    {index}/10
                    {index === 0
                      ? " — Aucun effort"
                      : index === 10
                        ? " — Effort maximal"
                        : ""}
                  </option>
                )
              )}
            </select>
          </label>
        </div>

        <small className="rv-muted">
          Utilisez uniquement des informations de test
          dans cette version.
        </small>
      </details>

      {(report.level === "review" ||
        report.level === "check" ||
        followup.reviewed) && (
        <div className="rv-actions">
          <button
            className={
              followup.reviewed
                ? "rv-secondary"
                : "rv-primary"
            }
            onClick={() =>
              onChange({
                ...followup,
                reviewed: !followup.reviewed,
              })
            }
          >
            <CheckCircle2 size={18} />

            {followup.reviewed
              ? "Rouvrir la revue"
              : "Marquer le bilan comme consulté"}
          </button>
        </div>
      )}

      <small className="rv-muted">
        Ce bilan accompagne le suivi sportif.
        Il ne constitue ni un diagnostic médical
        ni une autorisation de reprise.
      </small>
    </section>
  );
}
