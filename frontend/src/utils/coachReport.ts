import type {
  SportSession,
  SessionSample,
  PhysiologicalState,
} from "../data/sessionModels";

import { getHitlCounts } from "./sessionReview";

export type CoachAnswer =
  | "unknown"
  | "no"
  | "yes";

export interface CoachFollowup {
  pain: CoachAnswer;
  unusualFatigue: CoachAnswer;
  perceivedEffort: number | null;
  reviewed: boolean;
}

export const emptyFollowup: CoachFollowup = {
  pain: "unknown",
  unusualFatigue: "unknown",
  perceivedEffort: null,
  reviewed: false,
};

type Metric = "heartRate" | "respiratoryRate";

function median(values: number[]) {
  if (!values.length) return null;

  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);

  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function signalMedian(
  samples: SessionSample[],
  metric: Metric
) {
  const values = samples
    .map((sample) => sample[metric])
    .filter(
      (value): value is number =>
        value !== null &&
        Number.isFinite(value) &&
        value > 0
    );

  if (
    values.length < 3 ||
    values.length / samples.length < 0.6
  ) {
    return null;
  }

  return median(values);
}

function samplesForState(
  session: SportSession,
  state: PhysiologicalState
) {
  if (
    state === "rest" &&
    session.initialRestConfirmed &&
    session.initialRestSeconds
  ) {
    return session.samples.filter(
      (sample) =>
        sample.minute * 60 < session.initialRestSeconds!
    );
  }

  return session.samples.filter(
    (sample) =>
      (
        state !== "effort" ||
        session.effortEndSecond === undefined ||
        sample.minute * 60 < session.effortEndSecond
      ) &&
      session.phases.some(
        (phase) =>
          phase.state === state &&
          sample.minute >= phase.startMinute &&
          sample.minute < phase.endMinute
      )
  );
}

export function buildCoachReport(
  session: SportSession,
  followup: CoachFollowup = emptyFollowup
) {
  const counts = getHitlCounts(session);
  const total = session.predictions.length;

  const rest = samplesForState(session, "rest");
  const effort = samplesForState(session, "effort");

  const restingHR = signalMedian(rest, "heartRate");
  const effortHR = signalMedian(effort, "heartRate");

  const restingBR = signalMedian(
    rest, "respiratoryRate"
  );

  const effortBR = signalMedian(
    effort, "respiratoryRate"
  );

  let effortText =
    "Les mesures disponibles ne permettent pas encore de comparer le repos et l'effort.";

  if (restingHR !== null && effortHR !== null) {
    const difference = effortHR - restingHR;

    effortText =
      `Pendant l'effort, la fréquence cardiaque médiane ` +
      `atteint ${effortHR.toFixed(0)} bpm, contre ` +
      `${restingHR.toFixed(0)} bpm au repos. `;

    if (difference > 0) {
      effortText +=
        "Cette augmentation traduit une sollicitation cardiovasculaire plus importante pendant l'activité.";
    } else {
      effortText +=
        "Les mesures ne montrent pas d'augmentation cardiaque entre le repos et l'effort. Les conditions de l'entraînement doivent être prises en compte.";
    }
  }

  if (restingBR !== null && effortBR !== null) {
    effortText +=
      ` La fréquence respiratoire médiane passe de ` +
      `${restingBR.toFixed(1)} à ` +
      `${effortBR.toFixed(1)} respirations par minute.`;
  }

  effortText +=
    " L'intensité de la séance doit être appréciée selon les capacités individuelles du sportif.";

  const recoveryPhases = session.phases
    .filter(
      (phase) =>
        phase.state === "recovery" &&
        session.phases.some(
          (p) =>
            p.state === "effort" &&
            p.endMinute <= phase.startMinute + 0.01
        )
    )
    .sort(
      (a, b) => a.startMinute - b.startMinute
    );

  const recovery =
    session.effortEndSecond !== undefined
      ? {
          startMinute: session.effortEndSecond / 60,
          endMinute: session.duration + 1 / 60,
        }
      : session.predictionSource === "live-model"
        ? undefined
        : recoveryPhases[recoveryPhases.length - 1];

  let recoveryText =
    "Les informations disponibles ne permettent pas encore d'évaluer la récupération après l'effort.";

  if (recovery) {
    const duration =
      recovery.endMinute - recovery.startMinute;

    const early = session.samples.filter(
      (sample) =>
        sample.minute >= recovery.startMinute &&
        sample.minute <
          recovery.startMinute + duration / 3
    );

    const late = session.samples.filter(
      (sample) =>
        sample.minute >=
          recovery.endMinute - duration / 3 &&
        sample.minute < recovery.endMinute
    );

    const startHR = signalMedian(
      early, "heartRate"
    );

    const endHR = signalMedian(
      late, "heartRate"
    );

    const startBR = signalMedian(
      early, "respiratoryRate"
    );

    const endBR = signalMedian(
      late, "respiratoryRate"
    );

    if (startHR !== null && endHR !== null) {
      const decrease = startHR - endHR;

      recoveryText =
        `Entre le début et la fin de la dernière période ` +
        `de récupération, la fréquence cardiaque médiane ` +
        `passe de ${startHR.toFixed(0)} à ` +
        `${endHR.toFixed(0)} bpm. `;

      if (decrease > 0) {
        recoveryText +=
          `Cette diminution de ${decrease.toFixed(0)} bpm ` +
          "indique une baisse de la sollicitation cardiaque après l'effort.";
      } else {
        recoveryText +=
          "Aucune diminution cardiaque n'est observée entre ces deux périodes. Le contexte de récupération mérite d'être considéré.";
      }
    }

    if (startBR !== null && endBR !== null) {
      recoveryText +=
        ` La fréquence respiratoire médiane passe de ` +
        `${startBR.toFixed(1)} à ` +
        `${endBR.toFixed(1)} respirations par minute.`;
    }
  }

  if (
    session.predictionSource === "live-model" &&
    session.effortEndSecond === undefined
  ) {
    recoveryText +=
      " La fin réelle de l'effort n'est pas renseignée.";
  }

  recoveryText +=
    " Les prochaines séances comparables permettront de mieux apprécier les habitudes de récupération du sportif.";

  if (session.predictionSource === "live-model") {
    effortText +=
      " Les périodes d'effort sont estimées à partir des mesures.";
  }

  if (session.effortEndSecond !== undefined) {
    recoveryText +=
      " Cette observation repose sur la fin d'effort renseignée pour la séance.";
  }

  let fatigueText =
    "Le niveau de fatigue ne peut pas être déterminé à partir des seules mesures disponibles.";

  if (followup.unusualFatigue === "yes") {
    fatigueText =
      "Une fatigue inhabituelle a été signalée. Il est conseillé de vérifier la charge récente et les conditions de récupération.";
  } else if (followup.unusualFatigue === "no") {
    fatigueText =
      "Aucune fatigue inhabituelle n'a été signalée par le sportif.";
  }

  if (followup.perceivedEffort !== null) {
    fatigueText +=
      ` La difficulté ressentie de la séance est de ` +
      `${followup.perceivedEffort}/10.`;
  }

  let injuryText =
    "Le risque individuel de blessure reste indéterminé. Le suivi des charges d'entraînement, des douleurs, des antécédents et de la récupération permettra de mieux orienter la prévention.";

  if (followup.pain === "yes") {
    injuryText =
      "Une douleur ou une gêne a été signalée. Une attention particulière est recommandée avant la prochaine sollicitation. Un avis professionnel peut être nécessaire.";
  }

  let level:
    | "accept"
    | "check"
    | "review"
    | "neutral" = "neutral";

  let title = "Bilan incomplet";

  let conclusion =
    "Les informations disponibles ne suffisent pas encore pour établir un bilan complet.";

  let recommendation =
    "Compléter les mesures et les informations de séance.";

  if (followup.pain === "yes") {
    level = "review";
    title = "Attention du coach recommandée";

    conclusion =
      "Une douleur ou une gêne signalée nécessite une attention particulière.";

    recommendation =
      "Échanger avec le sportif sur la gêne ressentie avant la prochaine sollicitation. Demander un avis professionnel si nécessaire.";

  } else if (counts.Review > 0) {
    level = "review";
    title = "Séance à examiner";

    conclusion =
      "Certaines périodes de l'entraînement restent difficiles à interpréter. Une revue globale de la séance est recommandée.";

    recommendation =
      "Vérifier le déroulement de l'entraînement, les conditions de récupération et le ressenti du sportif.";

  } else if (
    followup.unusualFatigue === "yes" ||
    (total > 0 && counts.Check > counts.Accept)
  ) {
    level = "check";
    title = "Suivi conseillé";

    conclusion =
      followup.unusualFatigue === "yes"
        ? "Une fatigue inhabituelle a été signalée."
        : "Certaines périodes de la séance méritent un complément de contexte, sans revue prioritaire.";

    recommendation =
      "Recueillir le ressenti du sportif et observer son évolution lors des prochaines séances.";

  } else if (total > 0) {
    level = "accept";
    title = "Suivi courant";

    conclusion =
      "Aucune revue prioritaire n'est demandée selon les critères actuellement disponibles.";

    recommendation =
      "Poursuivre le suivi habituel et comparer les prochaines séances.";
  }

  // Une couverture partielle ne doit jamais
  // etre assimilee a un bilan rassurant.
  if (session.analysisStatus === "partial") {
    if (level === "accept") {
      level = "check";
      title = "Suivi conseillé";
    }

    conclusion +=
      " Certaines périodes n'ont pas pu être analysées en raison de mesures manquantes.";

    recommendation +=
      " Vérifier la disponibilité des mesures.";
  }

  // Regle experimentale de tri :
  // activite estimee persistante apres la fin
  // declaree de l'effort.
  if (session.effortEndSecond !== undefined) {
    const afterEffort = session.predictions.filter(
      (p) => p.startSecond >= session.effortEndSecond!
    );

    const persistentEffort =
      afterEffort.length >= 3 &&
      afterEffort.every(
        (p) => p.predictedState === "effort"
      );

    if (
      persistentEffort &&
      followup.pain !== "yes"
    ) {
      level = "review";
      title = "Récupération à examiner";

      conclusion =
        "Après la fin d'effort renseignée, les mesures continuent d'être interprétées comme de l'effort. La distinction entre activité et récupération mérite une vérification.";

      recommendation =
        "Vérifier si le sportif a réellement arrêté l'effort, si la récupération était active et recueillir son ressenti.";
    }
  }

  if (
    total === 0 &&
    level === "neutral" &&
    session.analysisMessage
  ) {
    conclusion = session.analysisMessage;
  }

  return {
    level,
    title,
    conclusion,
    recommendation,
    effortText,
    recoveryText,
    fatigueText,
    injuryText,
  };
}
