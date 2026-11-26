import {
  useEffect,
  useRef,
  useState,
} from "react";

import type { FormEvent } from "react";

import {
  Activity,
  RefreshCw,
} from "lucide-react";

import type { SportSession } from "../data/sessionModels";

import {
  analyzeSportSession,
  makeAnalysisContext,
} from "../utils/sessionAnalysis";

interface ContextProps {
  confirmed: boolean;
  restMinutes: string;
  endMinute: string;
  onConfirmed: (value: boolean) => void;
  onRestMinutes: (value: string) => void;
  onEndMinute: (value: string) => void;
}

export function AnalysisContextFields({
  confirmed,
  restMinutes,
  endMinute,
  onConfirmed,
  onRestMinutes,
  onEndMinute,
}: ContextProps) {
  return (
    <div className="rv-analysis-context">
      <h3>Informations sur l'entraînement</h3>

      <label className="rv-confirm-rest">
        <input
          type="checkbox"
          checked={confirmed}
          onChange={(e) =>
            onConfirmed(e.target.checked)
          }
        />

        <span>
          L'enregistrement commence par une période
          de repos du sportif.
        </span>
      </label>

      {confirmed && (
        <label>
          Durée du repos initial (minutes)

          <input
            type="number"
            min="4.59"
            step="any"
            value={restMinutes}
            onChange={(e) =>
              onRestMinutes(e.target.value)
            }
            required
          />
        </label>
      )}

      <label>
        Fin de l'effort (minute de la séance,
        facultatif)

        <input
          type="number"
          min="0"
          step="any"
          value={endMinute}
          placeholder="Ex. 25"
          onChange={(e) =>
            onEndMinute(e.target.value)
          }
        />
      </label>

      <small className="rv-muted">
        Le repos initial permet de personnaliser
        l'analyse. La fin réelle de l'effort aide
        à interpréter la récupération.
      </small>
    </div>
  );
}

interface Props {
  session: SportSession;
  sport: string;
  onSave: (session: SportSession) => void;
  onBusyChange: (busy: boolean) => void;
}

export default function SessionAnalysis({
  session,
  sport,
  onSave,
  onBusyChange,
}: Props) {
  const [confirmed, setConfirmed] = useState(
    session.initialRestConfirmed ?? false
  );

  const [restMinutes, setRestMinutes] = useState(
    String((session.initialRestSeconds || 300) / 60)
  );

  const [endMinute, setEndMinute] = useState(
    session.effortEndSecond === undefined
      ? ""
      : String(session.effortEndSecond / 60)
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const request = useRef<AbortController | null>(
    null
  );

  useEffect(() => {
    return () => request.current?.abort();
  }, []);

  async function submit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();
    if (loading) return;

    setError("");

    try {
      const context = makeAnalysisContext(
        confirmed,
        restMinutes,
        endMinute,
        session.duration
      );

      const controller = new AbortController();
      request.current = controller;

      setLoading(true);
      onBusyChange(true);

      const updated = await analyzeSportSession(
        session,
        sport,
        context,
        controller.signal
      );

      if (!controller.signal.aborted) {
        setLoading(false);
        onBusyChange(false);
        onSave(updated);
      }

    } catch (err) {
      setLoading(false);
      onBusyChange(false);

      setError(
        err instanceof Error
          ? err.message
          : "Analyse indisponible."
      );
    }
  }

  return (
    <form className="rv-panel rv-form" onSubmit={submit}>
      <div className="rv-section-heading">
        <div>
          <h2>Analyser la séance</h2>
          <p>
            Obtenez automatiquement votre bilan sportif.
          </p>
        </div>

        <Activity size={24} />
      </div>

      {session.analysisMessage && (
        <div className="rv-notice">
          {session.analysisMessage}
        </div>
      )}

      <AnalysisContextFields
        confirmed={confirmed}
        restMinutes={restMinutes}
        endMinute={endMinute}
        onConfirmed={setConfirmed}
        onRestMinutes={setRestMinutes}
        onEndMinute={setEndMinute}
      />

      {error && (
        <p className="rv-error" role="alert">
          {error}
        </p>
      )}

      <button
        className="rv-primary"
        type="submit"
        disabled={loading || !confirmed}
      >
        <RefreshCw
          size={18}
          className={loading ? "rv-spinning" : ""}
        />

        {loading
          ? "RHYVA analyse votre séance..."
          : "Obtenir mon bilan"}
      </button>
    </form>
  );
}
