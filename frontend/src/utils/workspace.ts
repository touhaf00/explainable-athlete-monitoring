import { sportdbAthletes } from "../data/sportdbAthletes";
import type { SportAthlete } from "../data/sessionModels";

export const WORKSPACE_KEY = "rhyva-v14-workspace";

export function loadWorkspace(): SportAthlete[] {
  try {
    const stored = localStorage.getItem(WORKSPACE_KEY);

    if (stored) {
      const parsed = JSON.parse(stored);

      if (
        Array.isArray(parsed.athletes) &&
        parsed.athletes.every(
          (a: SportAthlete) =>
            typeof a.id === "string" &&
            typeof a.name === "string" &&
            Array.isArray(a.sessions)
        )
      ) {
        return parsed.athletes;
      }
    }

    const previous = JSON.parse(
      localStorage.getItem(
        "rhyva-v12-coach-athletes"
      ) ?? '{"athletes":[]}'
    );

    return [
      ...sportdbAthletes,
      ...(Array.isArray(previous.athletes)
        ? previous.athletes.filter(
            (a: SportAthlete) => a.source === "coach"
          )
        : []),
    ];
  } catch {
    return [...sportdbAthletes];
  }
}

export function saveWorkspace(
  athletes: SportAthlete[]
) {
  localStorage.setItem(
    WORKSPACE_KEY,
    JSON.stringify({ athletes })
  );
}
