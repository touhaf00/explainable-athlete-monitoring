import type {
  SessionSample,
  SportSession,
} from "../data/sessionModels";

export function parseSessionCsv(
  content: string
): SessionSample[] {
  const lines = content
    .replace(/^\uFEFF/, "")
    .trim()
    .split(/\r?\n/)
    .filter((line) => line.trim());

  if (lines.length < 3) {
    throw new Error(
      "Le fichier doit contenir au moins deux mesures."
    );
  }

  if (lines.length > 10001) {
    throw new Error(
      "La maquette accepte jusqu'à 10 000 mesures par import."
    );
  }

  const separator = lines[0].includes(";")
    ? ";"
    : ",";

  const headers = lines[0]
    .split(separator)
    .map((value) => value.trim().toLowerCase());

  const timeIndex = headers.indexOf("time_seconds");
  const hrIndex = headers.indexOf("heart_rate");
  const brIndex = headers.indexOf("respiratory_rate");

  if (
    timeIndex < 0 ||
    hrIndex < 0 ||
    brIndex < 0
  ) {
    throw new Error(
      "Colonnes attendues : time_seconds, heart_rate, respiratory_rate."
    );
  }

  function parseNumber(value: string) {
    const normalized = value.trim();

    if (!normalized) return NaN;

    return Number(
      separator === ";"
        ? normalized.replace(",", ".")
        : normalized
    );
  }

  function readMetric(
    value: string,
    line: number
  ): number | null {
    const normalized = value.trim().toLowerCase();

    if (
      !normalized ||
      ["na", "nan", "null"].includes(normalized)
    ) {
      return null;
    }

    const number = parseNumber(value);

    if (!Number.isFinite(number) || number < 0) {
      throw new Error(
        `Mesure invalide à la ligne ${line}.`
      );
    }

    return number === 0 ? null : number;
  }

  let previousTime = -1;

  const samples = lines.slice(1).map((line, index) => {
    const cells = line.split(separator);

    if (cells.length !== headers.length) {
      throw new Error(
        `Nombre de colonnes incorrect à la ligne ${index + 2}.`
      );
    }

    const time = parseNumber(cells[timeIndex]);

    if (
      !Number.isFinite(time) ||
      time < 0 ||
      time <= previousTime
    ) {
      throw new Error(
        `Temps invalide ou non croissant à la ligne ${index + 2}.`
      );
    }

    previousTime = time;

    return {
      minute: time / 60,
      heartRate: readMetric(
        cells[hrIndex],
        index + 2
      ),
      respiratoryRate: readMetric(
        cells[brIndex],
        index + 2
      ),
    };
  });

  if (
    !samples.some(
      (s) => s.heartRate !== null ||
        s.respiratoryRate !== null
    )
  ) {
    throw new Error(
      "Aucune mesure physiologique exploitable."
    );
  }

  const origin = samples[0].minute;

  return samples.map((sample) => ({
    ...sample,
    minute: sample.minute - origin,
  }));
}

export function createSession(
  title: string,
  date: string,
  samples: SessionSample[] = [],
  existing?: SportSession
): SportSession {
  const count = samples.length;

  return {
    id: existing?.id ?? `session-${crypto.randomUUID()}`,

    title: title.trim(),
    date,
    inputSource: count ? "csv" : "manual",

    source: "coach",

    duration: count
      ? samples[count - 1].minute
      : 0,

    rawSampleCount: count,

    aggregationSeconds: 0,
    aggregationMethod: "none",

    phaseSource: "none",
    phases: [],
    samples,

    heartRateCoverage: count
      ? samples.filter(
          (s) => s.heartRate !== null
        ).length / count
      : 0,

    respiratoryRateCoverage: count
      ? samples.filter(
          (s) => s.respiratoryRate !== null
        ).length / count
      : 0,

    predictionSource: "none",
    predictions: [],
  };
}

export function describeSignal(
  samples: SessionSample[],
  metric: "heartRate" | "respiratoryRate"
) {
  const values = samples
    .map((sample) => sample[metric])
    .filter(
      (value): value is number =>
        value !== null && Number.isFinite(value)
    );

  if (!values.length) return null;

  return {
    min: Math.min(...values),
    max: Math.max(...values),
    mean:
      values.reduce((a, b) => a + b, 0) /
      values.length,
  };
}
