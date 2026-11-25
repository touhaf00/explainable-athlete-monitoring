
import { useState } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  HeartPulse,
  LayoutDashboard,
  Search,
  Users
} from "lucide-react";

import {
  athletes,
  type Athlete
} from "./data/athletes";

import "./styles/app.css";
import rhyvaLogo from "./assets/rhyva-logo.png";

type Screen = "dashboard" | "athletes";

function HeartChart({ values }: { values: number[] }) {
  const points = values.map((value, index) => {
    const x = 20 + (index / (values.length - 1)) * 560;
    const y = 190 - (value - 60) * 1.1;
    return `${x},${y}`;
  }).join(" ");

  return (
    <div className="chart-container">
      <svg
        viewBox="0 0 600 220"
        role="img"
        aria-label="Courbe fictive de fréquence cardiaque"
        preserveAspectRatio="none"
      >
        {[60, 110, 160].map((value) => (
          <line
            key={value}
            x1="20"
            x2="580"
            y1={190 - (value - 60) * 1.1}
            y2={190 - (value - 60) * 1.1}
            stroke="#e6ebf2"
            strokeDasharray="5 6"
          />
        ))}

        <polyline
          points={points}
          fill="none"
          stroke="#f17f6f"
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      <div className="chart-caption">
        <span>Début</span>
        <span>Effort</span>
        <span>Récupération</span>
        <span>Fin</span>
      </div>
    </div>
  );
}

function App() {
  const [screen, setScreen] = useState<Screen>("dashboard");
  const [selectedAthlete, setSelectedAthlete] =
    useState<Athlete | null>(null);

  const [search, setSearch] = useState("");
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [draftNote, setDraftNote] = useState("");
  const [reviewed, setReviewed] = useState<string[]>([]);
  const [feedback, setFeedback] = useState("");

  const attentionCount = athletes.filter(
    (athlete) =>
      athlete.status !== "normal" &&
      !reviewed.includes(athlete.id)
  ).length;

  const visibleAthletes = athletes.filter((athlete) =>
    athlete.name.toLowerCase().includes(search.toLowerCase()) ||
    athlete.sport.toLowerCase().includes(search.toLowerCase())
  );

  function navigate(destination: Screen) {
    setScreen(destination);
    setSearch("");
    setSelectedAthlete(null);
    setFeedback("");
  }

  function openAthlete(athlete: Athlete) {
    setSelectedAthlete(athlete);
    setDraftNote(notes[athlete.id] ?? "");
    setFeedback("");
  }

  function saveNote() {
    if (!selectedAthlete) return;

    setNotes((previous) => ({
      ...previous,
      [selectedAthlete.id]: draftNote.trim()
    }));

    setFeedback("Note enregistrée pour cette démonstration.");
  }

  function markReviewed() {
    if (!selectedAthlete) return;

    setReviewed((previous) =>
      previous.includes(selectedAthlete.id)
        ? previous
        : [...previous, selectedAthlete.id]
    );

    setFeedback("Séance marquée comme examinée.");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <img
              src={rhyvaLogo}
              alt="Logo Rhyva"
              className="brand-logo"
          />
          <strong>Rhyva</strong>
        </div>

        <span className="sidebar-label">ESPACE COACH</span>

        <nav className="navigation" aria-label="Navigation principale">
          <button
              className={
                screen === "dashboard" && !selectedAthlete
                    ? "nav-item active"
                    : "nav-item"
              }
              onClick={() => navigate("dashboard")}
          >
            <LayoutDashboard size={19}/>
            Tableau de bord
          </button>

          <button
              className={
                screen === "athletes" && !selectedAthlete
                    ? "nav-item active"
                    : "nav-item"
              }
              onClick={() => navigate("athletes")}
          >
            <Users size={19}/>
            Mes sportifs
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="coach-avatar">C</div>
          <div>
            <strong>Espace entraîneur</strong>
            <small>Prototype Rhyva</small>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <span>Espace coach / Rhyva</span>
          <span className="demo-badge">Prototype · Données fictives</span>
        </header>

        {selectedAthlete ? (
          <>
            <button
              className="back-button"
              onClick={() => setSelectedAthlete(null)}
            >
              <ArrowLeft size={17} />
              Retour aux sportifs
            </button>

            <section className="page-heading">
              <span className="eyebrow">ANALYSE INDIVIDUELLE</span>
              <h1>{selectedAthlete.name}</h1>
              <p>
                {selectedAthlete.sport} · Dernière séance :
                {" "}{selectedAthlete.duration} minutes
              </p>
            </section>

            <div className="details-grid">
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Analyse de la séance</h2>
                    <p>Évolution illustrative des mesures</p>
                  </div>
                  <HeartPulse size={22} className="accent-icon" />
                </div>

                <HeartChart
                  values={selectedAthlete.heartRateCurve}
                />

                <div className="phase-row">
                  <span>Repos</span>
                  <span>Effort</span>
                  <span>Récupération</span>
                </div>

                <div className="measure-grid">
                  <div>
                    <small>Durée de séance</small>
                    <strong>{selectedAthlete.duration} min</strong>
                  </div>
                  <div>
                    <small>FC après 8 min</small>
                    <strong>{selectedAthlete.heartRate} bpm</strong>
                  </div>
                  <div>
                    <small>Référence personnelle</small>
                    <strong>{selectedAthlete.personalRange}</strong>
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Ce que montre Rhyva</h2>
                    <p>Explication illustrative</p>
                  </div>
                </div>

                <span className={`status ${
                  reviewed.includes(selectedAthlete.id)
                    ? "status-normal"
                    : `status-${selectedAthlete.status}`
                }`}>
                  {reviewed.includes(selectedAthlete.id)
                    ? "Séance examinée"
                    : selectedAthlete.statusLabel}
                </span>

                <h3 className="observation">
                  {selectedAthlete.observation}
                </h3>

                <p className="explanation">
                  {selectedAthlete.explanation}
                </p>

                <div className="coach-decision">
                  <strong>Le coach reste décisionnaire</strong>
                  <p>
                    Rhyva aide à lire les données.
                    Le coach interprète les résultats et décide
                    des éventuelles adaptations.
                  </p>
                </div>

                <button
                  className="primary-button"
                  onClick={markReviewed}
                >
                  <CheckCircle2 size={18} />
                  J'ai examiné cette séance
                </button>
              </section>
            </div>

            <section className="panel note-panel">
              <h2>Note du coach</h2>
              <p>
                Ajoute une observation pour cette séance fictive.
              </p>

              <textarea
                value={draftNote}
                onChange={(event) =>
                  setDraftNote(event.target.value)}
                placeholder="Exemple : demander au sportif son ressenti après la séance..."
                maxLength={500}
                rows={4}
              />

              <button
                className="secondary-button"
                onClick={saveNote}
              >
                Enregistrer la note
              </button>

              <p className="feedback" role="status">
                {feedback}
              </p>

              <small className="helper-text">
                Les modifications restent en mémoire pendant
                cette session. Aucune donnée n'est envoyée
                à un serveur.
              </small>
            </section>
          </>
        ) : (
          <>
            <section className="page-heading">
              <span className="eyebrow">
                {screen === "dashboard"
                  ? "VUE D'ENSEMBLE"
                  : "MES SPORTIFS"}
              </span>

              <h1>
                {screen === "dashboard"
                  ? "Bienvenue sur Rhyva"
                  : "Votre groupe d'entraînement"}
              </h1>

              <p>
                {screen === "dashboard"
                  ? "Une lecture plus simple des séances et de la récupération de vos sportifs."
                  : "Consultez les profils et examinez les dernières séances."}
              </p>
            </section>

            {screen === "dashboard" && (
              <div className="metrics-grid">
                <div className="metric-card">
                  <Users size={22} />
                  <small>Sportifs suivis</small>
                  <strong>{athletes.length}</strong>
                  <span>Profils de démonstration</span>
                </div>

                <div className="metric-card">
                  <Activity size={22} />
                  <small>Séances disponibles</small>
                  <strong>{athletes.length}</strong>
                  <span>Exemples fictifs</span>
                </div>

                <div className="metric-card">
                  <ClipboardList size={22} />
                  <small>À consulter</small>
                  <strong>{attentionCount}</strong>
                  <span>Suivis à examiner</span>
                </div>
              </div>
            )}

            <section className="panel athlete-panel">
              <div className="panel-heading">
                <div>
                  <h2>Les sportifs</h2>
                  <p>
                    Sélectionnez un profil pour consulter
                    sa dernière séance.
                  </p>
                </div>
              </div>

              {screen === "athletes" && (
                <label className="search-box">
                  <Search size={19} />
                  <input
                    type="search"
                    placeholder="Rechercher un sportif..."
                    value={search}
                    onChange={(event) =>
                      setSearch(event.target.value)}
                  />
                </label>
              )}

              {visibleAthletes.map((athlete) => (
                <button
                  key={athlete.id}
                  className="athlete-row"
                  onClick={() => openAthlete(athlete)}
                >
                  <span className="athlete-avatar">
                    {athlete.initials}
                  </span>

                  <span className="athlete-info">
                    <strong>{athlete.name}</strong>
                    <small>
                      {athlete.sport} · {athlete.duration} min
                    </small>
                  </span>

                  <span
                    className={
                      "status " +
                      (reviewed.includes(athlete.id)
                        ? "status-normal"
                        : `status-${athlete.status}`)
                    }
                  >
                    {reviewed.includes(athlete.id)
                      ? "Examiné"
                      : athlete.statusLabel}
                  </span>

                  <ArrowRight size={18} />
                </button>
              ))}

              {visibleAthletes.length === 0 && (
                <p>Aucun sportif ne correspond à la recherche.</p>
              )}
            </section>
          </>
        )}

        <footer className="app-footer">
          Rhyva · Prototype de recherche utilisateur ·
          Pas de diagnostic médical
        </footer>
      </main>
    </div>
  );
}

export default App;
