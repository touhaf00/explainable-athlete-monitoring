import {
  analyzeSportSession,
  makeAnalysisContext,
} from "../utils/sessionAnalysis";

import {
  AnalysisContextFields,
} from "./SessionAnalysis";

import { useState, useEffect, useRef } from "react";
import type { FormEvent } from "react";

import {
  FileUp,
  Watch,
  Bluetooth,
  CalendarPlus,
  X,
} from "lucide-react";

import type {
  SportSession,
  SessionSample,
} from "../data/sessionModels";

import {
  parseSessionCsv,
  createSession,
} from "../utils/sessionImport";

interface Props {
  sport: string;
  existing?: SportSession;
  onSave: (session: SportSession) => void;
  onClose: () => void;
}

type Source = "csv" | "watch" | "sensor" | "manual";

export default function SessionIntake({
  sport,
  existing,
  onSave,
  onClose,
}: Props) {
  const [title, setTitle] = useState(
    existing?.title ?? ""
  );

  const [date, setDate] = useState(
    existing?.date ??
      new Date().toLocaleDateString("sv-SE")
  );

  const [source, setSource] =
    useState<Source>("csv");

  const [samples, setSamples] =
    useState<SessionSample[]>([]);

  const [filename, setFilename] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const [confirmed, setConfirmed] = useState(false);
  const [restMinutes, setRestMinutes] = useState("5");
  const [endMinute, setEndMinute] = useState("");

  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => request.current?.abort();
  }, []);

  async function readFile(file?: File) {
    if (!file) return;

    setLoading(true);
    setError("");
    setSamples([]);
    setFilename("");

    try {
      if (file.size > 512 * 1024) {
        throw new Error(
          "Fichier trop volumineux pour le prototype (512 Ko maximum)."
        );
      }

      const parsed = parseSessionCsv(
        await file.text()
      );

      setSamples(parsed);
      setFilename(file.name);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Impossible de lire le fichier."
      );
    } finally {
      setLoading(false);
    }
  }

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (loading) return;

    if (!title.trim() || !date) return;

    if (
      source !== "csv" &&
      source !== "manual"
    ) return;

    if (source === "csv" && !samples.length) {
      setError("Sélectionnez un fichier CSV valide.");
      return;
    }

    const session = createSession(
      title,
      date,
      source === "csv" ? samples : [],
      existing
    );

    if (source !== "csv") {
      onSave(session);
      return;
    }

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

      const analyzed = await analyzeSportSession(
        session,
        sport,
        context,
        controller.signal
      );

      if (!controller.signal.aborted) {
        setLoading(false);
        onSave(analyzed);
      }

    } catch (err) {
      setLoading(false);

      setError(
        err instanceof Error
          ? err.message
          : "Analyse indisponible."
      );
    }
  }

  function downloadTemplate() {
    const csv = [
      "time_seconds,heart_rate,respiratory_rate",
      "0,78,16",
      "15,82,17",
      "30,90,19",
      "45,105,22",
      "60,112,25",
      "75,102,21",
      "90,94,18",
    ].join("\n");

    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv" })
    );

    const link = document.createElement("a");

    link.href = url;
    link.download = "rhyva-exemple-test.csv";
    link.click();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const sources = [
    {
      id: "csv",
      label: "Importer un fichier",
      description: "CSV de mesures physiologiques",
      icon: FileUp,
    },
    {
      id: "watch",
      label: "Montre connectée",
      description: "Connexion à intégrer",
      icon: Watch,
    },
    {
      id: "sensor",
      label: "Capteur sportif",
      description: "Connexion à intégrer",
      icon: Bluetooth,
    },
    {
      id: "manual",
      label: "Préparer une séance",
      description: "Sans mesures pour le moment",
      icon: CalendarPlus,
    },
  ] as const;

  return (
    <form className="rv-panel rv-form" onSubmit={save}>
      <div className="rv-section-heading">
        <div>
          <h2>
            {existing
              ? "Ajouter les mesures"
              : "Nouvelle séance"}
          </h2>
          <p>Préparez ou importez votre entraînement.</p>
        </div>

        <button
          type="button"
          className="rv-icon-button"
          aria-label="Fermer"
          onClick={onClose}
        >
          <X size={20} />
        </button>
      </div>

      <div className="rv-form-grid">
        <label>
          Titre de la séance
          <input
            required
            maxLength={100}
            value={title}
            placeholder="Ex. Entraînement d'endurance"
            onChange={(e) => setTitle(e.target.value)}
          />
        </label>

        <label>
          Date
          <input
            required
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </label>
      </div>

      <h3>Comment récupérer les mesures ?</h3>

      <div className="rv-source-grid">
        {sources.map(({ id, label, description, icon: Icon }) => (
          <button
            type="button"
            key={id}
            className={
              "rv-source" +
              (source === id ? " rv-selected" : "")
            }
            aria-pressed={source === id}
            onClick={() => setSource(id)}
          >
            <Icon size={24} />
            <strong>{label}</strong>
            <small>{description}</small>
          </button>
        ))}
      </div>

      {source === "csv" && (
        <div className="rv-upload">
          <FileUp size={28} />

          <strong>Importer vos mesures</strong>

          <input
            type="file"
            accept=".csv,text/csv"
            disabled={loading}
            onChange={(e) =>
              void readFile(e.target.files?.[0])
            }
          />

          {loading && <p>Lecture du fichier...</p>}

          {samples.length > 0 && (
            <p>
              {filename} · {samples.length} mesures
            </p>
          )}

          <button
            type="button"
            className="rv-secondary"
            onClick={downloadTemplate}
          >
            Télécharger un CSV de test
          </button>

          <small>
            Format : time_seconds, heart_rate,
            respiratory_rate.
            Utilisez uniquement des données de test.
          </small>
        </div>
      )}

      {source === "csv" && samples.length > 0 && (
        <AnalysisContextFields
          confirmed={confirmed}
          restMinutes={restMinutes}
          endMinute={endMinute}
          onConfirmed={setConfirmed}
          onRestMinutes={setRestMinutes}
          onEndMinute={setEndMinute}
        />
      )}

      {(source === "watch" || source === "sensor") && (
        <div className="rv-notice">
          La connexion directe à cet appareil
          n'est pas encore disponible.

          Exportez ses mesures au format CSV
          pour les importer dans RHYVA.

          <button
            type="button"
            className="rv-secondary"
            onClick={() => setSource("csv")}
          >
            Importer un CSV
          </button>
        </div>
      )}

      {error && (
        <p className="rv-error" role="alert">
          {error}
        </p>
      )}

      <div className="rv-actions">
        <button
          type="button"
          className="rv-secondary"
          onClick={onClose}
        >
          Annuler
        </button>

        <button
          className="rv-primary"
          type="submit"
          disabled={
            loading ||
            source === "watch" ||
            source === "sensor"
          }
        >
          {loading
            ? "Traitement en cours..."
            : source === "csv" && confirmed
              ? "Enregistrer et analyser"
              : "Enregistrer la séance"}
        </button>
      </div>
    </form>
  );
}
