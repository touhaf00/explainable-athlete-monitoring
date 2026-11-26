export const reasons = {
  context: "Contexte de la séance manquant",
  sensor: "Qualité des mesures à vérifier",
  reference: "Récupération habituelle à comparer",
  classification: "Déroulement de l'entraînement à vérifier",
  other: "Autre raison"
} as const;

export type Reason = keyof typeof reasons;

export type Verdict =
  | "confirmed"
  | "corrected"
  | "insufficient";

export const verdictLabels: Record<Verdict, string> = {
  confirmed: "Observation maintenue",
  corrected: "Correction humaine proposée",
  insufficient: "Données insuffisantes"
};

export interface Report {
  reason: Reason;
  context: string;
  findings: string[];
  verdict?: Verdict;
  correction?: string;
}

export interface StudyState {
  reason: Reason | "";
  context: string;
  correction: string;
  reports: Report[];
  activeIndex: number | null;
}

export function emptyStudy(): StudyState {
  return {
    reason: "",
    context: "",
    correction: "",
    reports: [],
    activeIndex: null
  };
}

export function getStudyVerdict(
  study?: StudyState
): Verdict | undefined {
  if (!study || study.activeIndex === null) {
    return undefined;
  }

  return study.reports[study.activeIndex]?.verdict;
}

