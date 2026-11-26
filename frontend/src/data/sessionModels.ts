export type PhysiologicalState =
  | "rest"
  | "effort"
  | "recovery";

export interface SessionSample {
  minute: number;
  heartRate: number | null;
  respiratoryRate: number | null;
  heartRateCoverage?: number;
  respiratoryRateCoverage?: number;
}

export interface SessionPhase {
  state: PhysiologicalState;
  startMinute: number;
  endMinute: number;
  source: string;
}

export interface ModelPrediction {
  explanationFactors?: {
    feature: string;
    concept: string;
    contribution: number;
    similarity: number;
  }[];


  startSecond: number;
  endSecond: number;
  annotatedState: PhysiologicalState | null;
  predictedState: PhysiologicalState;
  confidence: number;
  correct: boolean | null;
  hitlDecision: "Accept" | "Check" | "Review";
  calibratedConfidence: number;
  calibratedReliability: number | null;
  acceptThreshold: number | "Infinity" | "-Infinity" | null;
  checkThreshold: number | "Infinity" | "-Infinity" | null;
  thresholdSource: string;

}

export interface SportSession {
  analysisStatus?:
    | "complete"
    | "partial"
    | "incomplete"
    | "error";

  analysisCoverage?: number;
  analysisMessage?: string;
  modelVersion?: string;

  initialRestConfirmed?: boolean;
  initialRestSeconds?: number;
  effortEndSecond?: number;


  title?: string;
  date?: string;
  inputSource?:
    | "csv"
    | "watch"
    | "sensor"
    | "manual"
    | "sportdb2";

  id: string;
  source: "SportDB2" | "coach";
  duration: number;
  rawSampleCount: number;
  aggregationSeconds: number;
  aggregationMethod: "median" | "none";

  phaseSource: "annotations" | "model" | "none";
  phases: SessionPhase[];

  samples: SessionSample[];

  heartRateCoverage: number;
  respiratoryRateCoverage: number;

  predictionSource: "loso-oof" | "live-model" | "none";
  predictions: ModelPrediction[];
}

export interface SportAthlete {
  id: string;
  name: string;
  initials: string;
  sport: string;

  source: "sportdb2" | "coach";
  fictionalIdentity: boolean;

  level?: string;
  objectives?: string;

  sessions: SportSession[];
}
