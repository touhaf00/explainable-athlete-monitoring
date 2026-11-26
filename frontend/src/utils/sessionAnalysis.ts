import type {
  SportSession,
  SessionPhase,
  ModelPrediction,
} from "../data/sessionModels";

const API_URL =
  import.meta.env.VITE_RHYVA_API_URL ??
  "http://127.0.0.1:8000";

export interface AnalysisContext {
  initialRestConfirmed: boolean;
  initialRestSeconds: number;
  effortEndSecond?: number;
}

interface ApiResult {
  status: "complete" | "partial" | "incomplete";
  message: string;
  analysisCoverage: number;
  predictions: ModelPrediction[];
  phases: SessionPhase[];
  modelVersion?: string;
}

function sportCode(sport: string) {
  const normalized = sport
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();

  if (
    normalized.includes("course") ||
    normalized.includes("running") ||
    normalized.includes("athletisme")
  ) return "RUN";

  if (
    normalized.includes("cyclisme") ||
    normalized.includes("velo")
  ) return "CYC";

  if (normalized.includes("americain")) {
    return "AMF";
  }

  if (normalized.includes("football")) {
    return "SOC";
  }

  return "OTHER";
}

export function makeAnalysisContext(
  confirmed: boolean,
  restMinutes: string,
  endMinute: string,
  duration: number
): AnalysisContext {
  const rest = confirmed
    ? Math.round(Number(restMinutes) * 60)
    : 0;

  const end = endMinute.trim()
    ? Number(endMinute) * 60
    : undefined;

  if (
    confirmed &&
    (
      !Number.isFinite(rest) ||
      rest < 275 ||
      rest >= duration * 60
    )
  ) {
    throw new Error(
      "Le repos initial doit durer au moins 4 min 35 s et être inclus dans la séance."
    );
  }

  if (
    end !== undefined &&
    (
      !Number.isFinite(end) ||
      end <= rest ||
      end >= duration * 60
    )
  ) {
    throw new Error(
      "La fin de l'effort doit se situer avant la fin de l'enregistrement et après le repos initial."
    );
  }

  return {
    initialRestConfirmed: confirmed,
    initialRestSeconds: rest,
    effortEndSecond: end,
  };
}

export async function analyzeSportSession(
  session: SportSession,
  sport: string,
  context: AnalysisContext,
  signal?: AbortSignal
): Promise<SportSession> {
  const base = {
    ...session,
    ...context,
  };

  if (!context.initialRestConfirmed) {
    return {
      ...base,
      analysisStatus: "incomplete",
      analysisMessage:
        "Une période de repos initiale confirmée est nécessaire pour personnaliser l'analyse.",
    };
  }

  const controller = new AbortController();

  const cancel = () => controller.abort();

  if (signal?.aborted) cancel();

  signal?.addEventListener("abort", cancel, {
    once: true,
  });

  const timeout = setTimeout(cancel, 120000);

  try {
    const response = await fetch(
      `${API_URL}/api/analyze`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        signal: controller.signal,

        body: JSON.stringify({
          sport: sportCode(sport),

          initialRestConfirmed:
            context.initialRestConfirmed,

          initialRestSeconds:
            context.initialRestSeconds,

          samples: session.samples.map((s) => ({
            second: s.minute * 60,
            heartRate: s.heartRate,
            respiratoryRate: s.respiratoryRate,
          })),
        }),
      }
    );

    const result = await response.json();

    if (!response.ok) {
      return {
        ...base,
        analysisStatus: "error",
        analysisMessage:
          typeof result.detail === "string"
            ? result.detail
            : "Les mesures ne peuvent pas être analysées.",
      };
    }

    const data = result as ApiResult;

    if (
      !["complete", "partial", "incomplete"]
        .includes(data.status) ||
      !Array.isArray(data.predictions) ||
      !Array.isArray(data.phases) ||
      !Number.isFinite(data.analysisCoverage) ||
      data.analysisCoverage < 0 ||
      data.analysisCoverage > 1
    ) {
      throw new Error("Réponse inattendue");
    }

    return {
      ...base,

      analysisStatus: data.status,
      analysisCoverage: data.analysisCoverage,
      analysisMessage: data.message,
      modelVersion: data.modelVersion,

      predictionSource: data.predictions.length
        ? "live-model"
        : "none",

      predictions: data.predictions,

      phaseSource: data.phases.length
        ? "model"
        : "none",

      phases: data.phases
        .map((phase) => ({
          ...phase,
          endMinute: Math.min(
            phase.endMinute,
            session.duration
          ),
        }))
        .filter(
          (phase) =>
            phase.endMinute > phase.startMinute
        ),
    };

  } catch {
    return {
      ...base,
      analysisStatus: "error",
      analysisMessage:
        "Le service d'analyse est momentanément indisponible. Vos mesures sont conservées.",
    };
  } finally {
    clearTimeout(timeout);

    signal?.removeEventListener("abort", cancel);
  }
}
