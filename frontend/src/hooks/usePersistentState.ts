import { useEffect, useState } from "react";

export function usePersistentState<T extends object>(
  key: string,
  initialValue: T
) {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);

      if (!stored) return initialValue;

      const parsed = JSON.parse(stored);

      if (
        parsed === null ||
        typeof parsed !== "object" ||
        Array.isArray(parsed)
      ) {
        return initialValue;
      }

      return parsed as T;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      console.warn("Stockage local RHYVA indisponible.");
    }
  }, [key, value]);

  return [value, setValue] as const;
}
