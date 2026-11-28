import { useState } from "react";

import {
  LayoutDashboard,
  Users,
  CalendarDays,
  MessageSquare,
  Watch,
  Activity,
  HeartPulse,
  ArrowLeft,
  ArrowRight,
  Plus,
  Trash2,
  FileUp,
  Bluetooth,
  CheckCircle2,
  Search,
  AlertCircle,
} from "lucide-react";

import type {
  SportAthlete,
  SportSession,
  ModelPrediction,
} from "./data/sessionModels";

import AthleteManager from "./components/AthleteManager";
import SessionIntake from "./components/SessionIntake";
import SessionAnalysis from "./components/SessionAnalysis";
import CoachInsight from "./components/CoachInsight";

import {
  buildCoachReport,
  emptyFollowup,
  type CoachFollowup,
} from "./utils/coachReport";
import PhysiologicalChart from "./components/PhysiologicalChart";
import AdaptiveReview from "./components/AdaptiveReview";
import ExplainabilityPanel from "./components/ExplainabilityPanel";

import { usePersistentState } from "./hooks/usePersistentState";

import {
  emptyStudy,
  type StudyState,
} from "./utils/adaptiveReview";

import {
  buildReviewTarget,
  buildWindowReviewTarget,
  getPredictionAtSample,
  getWindowSampleIndex,
  getHitlCounts,
  formatDuration,
  type SessionReviewTarget,
} from "./utils/sessionReview";

import {
  loadWorkspace,
  saveWorkspace,
} from "./utils/workspace";

import logo from "./assets/rhyva-logo.png";

import "./styles/app.css";
import "./styles/coach.css";

type Screen =
  | "dashboard"
  | "athletes"
  | "sessions"
  | "learning"
  | "devices";

type Decision = "agree" | "check" | "disagree";

const navigation = [
  ["dashboard", LayoutDashboard, "Accueil"],
  ["athletes", Users, "Mes sportifs"],
  ["sessions", CalendarDays, "Séances"],
  ["learning", MessageSquare, "Mes retours"],
  ["devices", Watch, "Appareils"],
] as const;

const humanChoices = [
  ["agree", "Je confirme"],
  ["check", "Approfondir l'analyse"],
  ["disagree", "Proposer une correction"],
] as const;

function sessionTitle(session: SportSession) {
  return session.title || "Séance d'entraînement";
}

function App() {
  const [screen, setScreen] =
    useState<Screen>("dashboard");

  const [athletes, setAthletes] =
    useState<SportAthlete[]>(loadWorkspace);

  const [profileId, setProfileId] =
    useState<string | null>(null);

  const [sessionId, setSessionId] =
    useState<string | null>(null);

  const [activeIndex, setActiveIndex] = useState(0);

  const [windowOverride, setWindowOverride] =
    useState<ModelPrediction>();

  const [intake, setIntake] =
    useState<"new" | "complete" | null>(null);

  const [voluntaryReview, setVoluntaryReview] =
    useState<string | null>(null);

  const [draftNote, setDraftNote] = useState("");
  const [feedback, setFeedback] = useState("");
  const [storageError, setStorageError] = useState("");

  const [analysisBusyId, setAnalysisBusyId] =
    useState<string | null>(null);


  const [followups, setFollowups] =
    usePersistentState<
      Partial<Record<string, CoachFollowup>>
    >("rhyva-v14-coach-followups", {});

  const [notes, setNotes] =
    usePersistentState<Record<string, string>>(
      "rhyva-v12-notes", {}
    );

  const [decisions, setDecisions] =
    usePersistentState<Partial<Record<string, Decision>>>(
      "rhyva-v13-window-decisions", {}
    );

  const [studies, setStudies] =
    usePersistentState<Partial<Record<string, StudyState>>>(
      "rhyva-v13-window-studies", {}
    );

  const sessions = athletes.flatMap(
    (athlete) => athlete.sessions.map(
      (session) => buildReviewTarget(athlete, session)
    )
  );

  const reviewTargets = sessions.flatMap(
    (target) => target.session.predictions.map(
      (prediction) =>
        buildWindowReviewTarget(target, prediction)
    )
  );

  const profile = athletes.find(
    (athlete) => athlete.id === profileId
  );

  const session = profile?.sessions.find(
    (item) => item.id === sessionId
  );

  const target = profile && session
    ? buildReviewTarget(profile, session)
    : undefined;

  const prediction = session
    ? windowOverride ??
      getPredictionAtSample(session, activeIndex)
    : undefined;

  const currentReview = target && prediction
    ? buildWindowReviewTarget(target, prediction)
    : undefined;

  const counts = session
    ? getHitlCounts(session)
    : { Accept: 0, Check: 0, Review: 0 };

  function getSessionStatus(item: SportSession) {
    const followup =
      followups[item.id] ?? emptyFollowup;

    const report =
      buildCoachReport(item, followup);

    if (
      followup.reviewed &&
      (
        report.level === "review" ||
        report.level === "check"
      )
    ) {
      return {
        label: "Bilan consulté",
        style: "neutral",
      };
    }

    return {
      label: report.title,
      style: report.level,
    };
  }

  const priorityGroups = athletes.flatMap((athlete) => {
    const pending = athlete.sessions.filter((item) => {
      const followup =
        followups[item.id] ?? emptyFollowup;

      return !followup.reviewed &&
        buildCoachReport(item, followup).level === "review";
    });

    if (!pending.length) return [];

    const latest = pending[pending.length - 1];

    return [{
      athlete,
      target: buildReviewTarget(athlete, latest),
      report: buildCoachReport(
        latest,
        followups[latest.id] ?? emptyFollowup
      ),
      pendingCount: pending.length,
    }];
  });

  const prioritySessions = priorityGroups.length;

  const monitoringCount = athletes.filter((athlete) =>
    !priorityGroups.some(
      (group) => group.athlete.id === athlete.id
    ) &&
    athlete.sessions.some((item) => {
      const followup =
        followups[item.id] ?? emptyFollowup;

      return !followup.reviewed &&
        buildCoachReport(item, followup).level === "check";
    })
  ).length;

  const feedbackTargets = reviewTargets.filter(
    (item) =>
      decisions[item.id] ||
      studies[item.id]?.reports.length
  );

  function navigate(destination: Screen) {
    setAnalysisBusyId(null);
    setScreen(destination);
    setProfileId(null);
    setSessionId(null);
    setIntake(null);
    setWindowOverride(undefined);
    setVoluntaryReview(null);
    setFeedback("");
  }

  function updateAthletes(next: SportAthlete[]) {
    try {
      saveWorkspace(next);
    } catch {
      setStorageError(
        "Stockage local indisponible ou saturé. La modification n'a pas été enregistrée."
      );
      return false;
    }

    setStorageError("");
    setAthletes(next);

    const remainingSessions = new Set(
      next.flatMap((athlete) =>
        athlete.sessions.map((s) => s.id)
      )
    );

    setFollowups((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(
          ([id]) => remainingSessions.has(id)
        )
      )
    );

    const allowed = new Set(
      next.flatMap(
        (athlete) => athlete.sessions.flatMap(
          (item) => [
            item.id,
            ...item.predictions.map(
              (p) => `${item.id}:${p.startSecond}-${p.endSecond}`
            ),
          ]
        )
      )
    );

    setNotes((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(
          ([id]) => allowed.has(id)
        )
      )
    );

    setDecisions((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(
          ([id]) => allowed.has(id)
        )
      )
    );

    setStudies((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(
          ([id]) => allowed.has(id)
        )
      )
    );

    return true;
  }

  function chooseWindow(
    item: SportSession,
    p: ModelPrediction
  ) {
    setWindowOverride(p);

    setActiveIndex(
      getWindowSampleIndex(item, p)
    );

    setVoluntaryReview(null);
  }

  function openSession(item: SessionReviewTarget) {
    setAnalysisBusyId(null);
    setProfileId(item.athleteId);
    setSessionId(item.session.id);
    setIntake(null);
    setFeedback("");
    setVoluntaryReview(null);

    setDraftNote(
      notes[item.session.id] ?? ""
    );

    const predictions = item.session.predictions;

    const preferred =
      predictions.some(
        (p) => p.hitlDecision === "Review"
      )
        ? "Review"
        : predictions.some(
            (p) => p.hitlDecision === "Accept"
          )
          ? "Accept"
          : "Check";

    const examples = predictions.filter(
      (p) => p.hitlDecision === preferred
    );

    const example =
      item.window ??
      examples[Math.floor(examples.length / 2)];

    if (example) {
      chooseWindow(item.session, example);
    } else {
      setWindowOverride(undefined);
      setActiveIndex(0);
    }
  }

  function openAthlete(id: string) {
    const athlete = athletes.find(
      (item) => item.id === id
    );

    if (!athlete) return;

    setScreen("athletes");
    setProfileId(id);
    setIntake(null);

    const last =
      athlete.sessions[athlete.sessions.length - 1];

    if (last) {
      openSession(
        buildReviewTarget(athlete, last)
      );
    } else {
      setSessionId(null);
      setWindowOverride(undefined);
      setActiveIndex(0);
      setDraftNote("");
      setFeedback("");
    }
  }

  function addSession(item: SportSession) {
    if (!profile) return;

    const exists = profile.sessions.some(
      (s) => s.id === item.id
    );

    const updated: SportAthlete = {
      ...profile,
      sessions: exists
        ? profile.sessions.map(
            (s) => s.id === item.id ? item : s
          )
        : [...profile.sessions, item],
    };

    if (
      updateAthletes(
        athletes.map(
          (a) => a.id === profile.id ? updated : a
        )
      )
    ) {
      openSession(
        buildReviewTarget(updated, item)
      );
    }
  }

  function deleteSession() {
    if (!profile || !session) return;

    if (!window.confirm(
      "Supprimer cette séance, ses mesures et ses retours ?"
    )) return;

    if (
      updateAthletes(
        athletes.map((a) =>
          a.id === profile.id
            ? {
                ...a,
                sessions: a.sessions.filter(
                  (s) => s.id !== session.id
                ),
              }
            : a
        )
      )
    ) {
      setSessionId(null);
      setIntake(null);
      setWindowOverride(undefined);
      setDraftNote("");
      setFeedback("");
    }
  }

  function submitDecision(decision: Decision) {
    if (!currentReview) return;

    const id = currentReview.id;

    if (decisions[id] !== decision) {
      setStudies((previous) => ({
        ...previous,
        [id]: {
          ...emptyStudy(),
          reports: previous[id]?.reports ?? [],
        },
      }));
    }

    setDecisions((previous) => ({
      ...previous,
      [id]: decision,
    }));
  }

  function resetFeedback() {
    if (!window.confirm(
      "Effacer les notes et les retours du coach ?"
    )) return;

    setNotes({});
    setDecisions({});
    setStudies({});
    setFollowups({});
    navigate("learning");
  }

  function saveNote() {
    if (!session) return;

    setNotes((previous) => ({
      ...previous,
      [session.id]: draftNote.trim(),
    }));

    setFeedback("Note enregistrée dans la maquette.");
  }

  function graphChange(index: number) {
    setActiveIndex(index);
    setWindowOverride(undefined);
    setVoluntaryReview(null);
  }

  return (
    <div className="rv-app">
      <aside className="rv-sidebar">
        <div className="rv-brand">
          <img src={logo} alt="RHYVA" />
          <div>
            <strong>RHYVA</strong>
            <small>Espace entraîneur</small>
          </div>
        </div>

        <nav aria-label="Navigation principale">
          {navigation.map(([id, Icon, label]) => (
            <button
              key={id}
              className={
                screen === id ? "rv-nav-active" : ""
              }
              onClick={() => navigate(id)}
            >
              <Icon size={20} />
              {label}
            </button>
          ))}
        </nav>

        <div className="rv-sidebar-bottom">
          <div className="rv-avatar">C</div>

          <div>
            <strong>Espace coach</strong>
            <small>Version de démonstration</small>
          </div>
        </div>
      </aside>

      <main className="rv-main">
        <header className="rv-topbar">
          <span>Votre espace de suivi sportif</span>
          <small>Prototype RHYVA</small>
        </header>

        {storageError && (
          <p className="rv-error" role="alert">
            {storageError}
          </p>
        )}

        {profile ? (
          <>
            <button
              className="rv-back"
              onClick={() => navigate("athletes")}
            >
              <ArrowLeft size={18} />
              Retour aux sportifs
            </button>

            <section className="rv-profile-header">
              <div className="rv-avatar rv-avatar-large">
                {profile.initials}
              </div>

              <div className="rv-profile-info">
                <h1>{profile.name}</h1>

                <p>
                  {profile.sport}
                  {profile.level
                    ? ` · ${profile.level}`
                    : ""}
                </p>

                {profile.objectives && (
                  <small>{profile.objectives}</small>
                )}
              </div>

              <button
                className="rv-primary"
                onClick={() => setIntake("new")}
              >
                <Plus size={18} />
                Ajouter une séance
              </button>
            </section>

            {intake && (
              <SessionIntake
                sport={profile.sport}
                key={intake === "complete"
                  ? session?.id
                  : "new-session"}
                existing={
                  intake === "complete"
                    ? session
                    : undefined
                }
                onSave={addSession}
                onClose={() => setIntake(null)}
              />
            )}

            <section className="rv-panel">
              <div className="rv-section-heading">
                <div>
                  <h2>Historique des séances</h2>
                  <p>
                    {profile.sessions.length}
                    {" "}séance(s) enregistrée(s)
                  </p>
                </div>
                <CalendarDays size={23} />
              </div>

              {!profile.sessions.length && (
                <div className="rv-empty">
                  <Activity size={30} />
                  <strong>Aucune séance pour le moment</strong>
                  <p>
                    Ajoutez un entraînement pour commencer.
                  </p>

                  <button
                    className="rv-primary"
                    onClick={() => setIntake("new")}
                  >
                    Créer une première séance
                  </button>
                </div>
              )}

              <div className="rv-list">
                {profile.sessions.map((item) => {
                  const status = getSessionStatus(item);

                  return (
                    <button
                      className={
                        "rv-list-row" +
                        (item.id === sessionId
                          ? " rv-selected"
                          : "")
                      }
                      key={item.id}
                      onClick={() =>
                        openSession(
                          buildReviewTarget(profile, item)
                        )
                      }
                    >
                      <Activity size={23} />

                      <span className="rv-list-info">
                        <strong>{sessionTitle(item)}</strong>
                        <small>
                          {item.date ?? "Date non renseignée"}
                          {" · "}
                          {item.samples.length
                            ? formatDuration(item.duration)
                            : "À compléter"}
                        </small>
                      </span>

                      <span className={
                        `rv-tag rv-${status.style}`
                      }>
                        {status.label}
                      </span>

                      <ArrowRight size={18} />
                    </button>
                  );
                })}
              </div>
            </section>

            {session && target && (
              <>
                <div className="rv-section-heading">
                  <div>
                    <h2>{sessionTitle(session)}</h2>

                    <p>
                      {session.samples.length
                        ? formatDuration(session.duration)
                        : "Mesures en attente"}
                    </p>
                  </div>

                  <button
                    className="rv-icon-button rv-danger"
                    aria-label="Supprimer cette séance"
                    onClick={deleteSession}
                  >
                    <Trash2 size={19} />
                  </button>
                </div>

                {session.source === "coach" &&
                  session.samples.length > 0 &&
                  session.predictions.length === 0 && (
                    <SessionAnalysis
                      key={session.id}
                      session={session}
                      sport={profile.sport}
                      onSave={addSession}
                      onBusyChange={(busy) =>
                        setAnalysisBusyId(
                          busy ? session.id : null
                        )
                      }
                    />
                  )}

                {analysisBusyId !== session.id && (
                <CoachInsight
                  key={session.id}
                  session={session}
                  followup={
                    followups[session.id] ?? emptyFollowup
                  }
                  onChange={(value) =>
                    setFollowups((previous) => ({
                      ...previous,
                      [session.id]: value,
                    }))
                  }
                />
                )}

                {!session.samples.length ? (
                  <section className="rv-panel rv-empty">
                    <FileUp size={32} />

                    <h3>Votre séance est prête</h3>

                    <p>
                      Ajoutez les mesures après
                      votre entraînement.
                    </p>

                    <button
                      className="rv-primary"
                      onClick={() => setIntake("complete")}
                    >
                      Importer les mesures
                    </button>
                  </section>
                ) : (
                  <>
                    <details className="rv-panel rv-details">
                      <summary>
                        Consulter les courbes de la séance
                      </summary>

                      <div className="rv-section-heading">
                        <div>
                          <h2>Évolution des mesures</h2>
                          <p>
                            Explorez le déroulement
                            de l'entraînement.
                          </p>
                        </div>

                        <HeartPulse size={23} />
                      </div>

                      <div className="rv-chart-grid">
                        {([
                          "heartRate",
                          "respiratoryRate",
                        ] as const).map((metric) => (
                          <PhysiologicalChart
                            key={`${session.id}-${metric}`}
                            athlete={target}
                            metric={metric}
                            activeIndex={activeIndex}
                            onIndexChange={graphChange}
                          />
                        ))}
                      </div>
                    </details>



                    {session.predictions.length > 0 && (
                      <details className="rv-panel rv-details">
                        <summary>
                          Examiner certains moments de la séance
                        </summary>
                        <div className="rv-section-heading">
                          <div>
                            <h2>Examen approfondi</h2>

                            <p>
                              Explorez les instants
                              importants de la séance.
                            </p>
                          </div>
                        </div>

                        <div className="rv-decision-grid">
                          {([
                            ["Accept", "Suivi courant", CheckCircle2],
                            ["Check", "À vérifier", Search],
                            ["Review", "À examiner", AlertCircle],
                          ] as const).map(([id, label, Icon]) => {
                            const examples =
                              session.predictions.filter(
                                (p) => p.hitlDecision === id
                              );

                            return (
                              <button
                                key={id}
                                disabled={!examples.length}
                                className={
                                  `rv-decision rv-${id.toLowerCase()}` +
                                  (prediction?.hitlDecision === id
                                    ? " rv-selected"
                                    : "")
                                }
                                onClick={() => {
                                  const current =
                                    prediction &&
                                    prediction.hitlDecision === id
                                      ? examples.findIndex(
                                          (p) =>
                                            p.startSecond ===
                                            prediction.startSecond
                                        )
                                      : -1;

                                  const next =
                                    examples[
                                      (current + 1) %
                                      examples.length
                                    ];

                                  if (next) {
                                    chooseWindow(session, next);
                                  }
                                }}
                              >
                                <Icon size={23} />

                                <strong>{counts[id]}</strong>

                                <span>{label}</span>

                                <small>
                                  {examples.length
                                    ? "Consulter le passage"
                                    : "Aucun instant"}
                                </small>
                              </button>
                            );
                          })}
                        </div>

                        {currentReview && prediction && (
                          <div className="rv-review-box">
                            <ExplainabilityPanel
                              athlete={currentReview}
                              activeIndex={activeIndex}
                              predictionOverride={prediction}
                            />

                            {prediction.hitlDecision === "Accept" &&
                              !decisions[currentReview.id] && (
                              <button
                                className="rv-secondary"
                                onClick={() =>
                                  setVoluntaryReview(
                                    voluntaryReview === currentReview.id
                                      ? null
                                      : currentReview.id
                                  )
                                }
                              >
                                Examiner volontairement
                              </button>
                            )}

                            {(prediction.hitlDecision !== "Accept" ||
                              voluntaryReview === currentReview.id ||
                              decisions[currentReview.id]) && (
                              <>
                                <h3>Votre interprétation</h3>

                                <p>
                                  Ce passage correspond à une période
                                  située vers {(
                                    prediction.startSecond / 60
                                  ).toFixed(1)} minutes.
                                </p>

                                <div className="rv-actions">
                                  {humanChoices.map(([id, label]) => (
                                    <button
                                      key={id}
                                      aria-pressed={
                                        decisions[currentReview.id] === id
                                      }
                                      className={
                                        decisions[currentReview.id] === id
                                          ? "rv-primary"
                                          : "rv-secondary"
                                      }
                                      onClick={() =>
                                        submitDecision(id)
                                      }
                                    >
                                      {label}
                                    </button>
                                  ))}
                                </div>

                                <AdaptiveReview
                                  key={currentReview.id}
                                  athlete={currentReview}
                                  decision={
                                    decisions[currentReview.id]
                                  }
                                  value={studies[currentReview.id]}
                                  onChange={(study) =>
                                    setStudies((previous) => ({
                                      ...previous,
                                      [currentReview.id]: study,
                                    }))
                                  }
                                />
                              </>
                            )}
                          </div>
                        )}
                      </details>
                    )}
                  </>
                )}

                <section className="rv-panel">
                  <h2>Mes observations</h2>

                  <p className="rv-muted">
                    Ressenti, conditions de séance,
                    éléments à revoir.
                  </p>

                  <textarea
                    className="rv-textarea"
                    rows={4}
                    maxLength={1000}
                    placeholder="Ajouter une observation..."
                    value={draftNote}
                    onChange={(e) =>
                      setDraftNote(e.target.value)
                    }
                  />

                  <button
                    className="rv-primary"
                    onClick={saveNote}
                  >
                    Enregistrer mon observation
                  </button>

                  <p role="status">{feedback}</p>
                </section>
              </>
            )}
          </>
        ) : screen === "athletes" ? (
          <AthleteManager
            athletes={athletes}
            onChange={(next) => {
              updateAthletes(next);
            }}
            onOpen={openAthlete}
          />
        ) : screen === "sessions" ? (
          <>
            <section className="rv-page-heading">
              <h1>Mes séances</h1>
              <p>Tous les entraînements de votre groupe.</p>
            </section>

            <section className="rv-panel">
              <div className="rv-section-heading">
                <h2>Historique général</h2>

                <button
                  className="rv-primary"
                  onClick={() => navigate("athletes")}
                >
                  <Plus size={18} />
                  Ajouter via un sportif
                </button>
              </div>

              {sessions.map((item) => {
                const status =
                  getSessionStatus(item.session);

                return (
                  <button
                    className="rv-list-row"
                    key={item.id}
                    onClick={() => openSession(item)}
                  >
                    <Activity size={22} />

                    <span className="rv-list-info">
                      <strong>{sessionTitle(item.session)}</strong>
                      <small>
                        {item.name} · {item.sport}
                      </small>
                      <small>
                        {item.session.date ??
                          "Date non renseignée"}
                      </small>
                    </span>

                    <span className={
                      `rv-tag rv-${status.style}`
                    }>
                      {status.label}
                    </span>

                    <ArrowRight size={18} />
                  </button>
                );
              })}

              {!sessions.length && (
                <p>Aucune séance enregistrée.</p>
              )}
            </section>
          </>
        ) : screen === "learning" ? (
          <>
            <section className="rv-page-heading">
              <h1>Mes retours</h1>

              <p>
                Retrouvez vos observations et
                interventions auprès de RHYVA.
              </p>
            </section>

            <section className="rv-panel">
              <h2>Mes interventions</h2>

              {!feedbackTargets.length && (
                <div className="rv-empty">
                  <MessageSquare size={30} />
                  <p>Aucune intervention enregistrée.</p>
                </div>
              )}

              {feedbackTargets.map((item) => (
                <button
                  className="rv-list-row"
                  key={item.id}
                  onClick={() => openSession(item)}
                >
                  <MessageSquare size={22} />

                  <span className="rv-list-info">
                    <strong>{item.name}</strong>

                    <small>
                      {sessionTitle(item.session)}
                      {" · "}
                      {((item.window?.startSecond ?? 0) / 60).toFixed(1)}
                      {" min"}
                    </small>

                    <small>
                      {studies[item.id]?.reports.length ?? 0}
                      {" "}analyse(s) complémentaire(s)
                    </small>
                  </span>

                  <ArrowRight size={18} />
                </button>
              ))}
            </section>

            <section className="rv-panel">
              <h2>Mes notes de séance</h2>

              {sessions.filter(
                (item) => notes[item.id]
              ).map((item) => (
                <button
                  className="rv-list-row"
                  key={item.id}
                  onClick={() => openSession(item)}
                >
                  <MessageSquare size={20} />

                  <span className="rv-list-info">
                    <strong>{item.name}</strong>
                    <small>{notes[item.id]}</small>
                  </span>
                </button>
              ))}

              {!sessions.some(
                (item) => notes[item.id]
              ) && <p>Aucune note enregistrée.</p>}
            </section>

            <section className="rv-panel">
              <h2>Bilans consultés</h2>

              {sessions.filter(
                (item) => followups[item.id]?.reviewed
              ).map((item) => (
                <button
                  key={item.id}
                  className="rv-list-row"
                  onClick={() => openSession(item)}
                >
                  <CheckCircle2 size={22} />

                  <span className="rv-list-info">
                    <strong>{item.name}</strong>
                    <small>{sessionTitle(item.session)}</small>
                    <small>Bilan consulté</small>
                  </span>

                  <ArrowRight size={18} />
                </button>
              ))}

              {!sessions.some(
                (item) => followups[item.id]?.reviewed
              ) && (
                <p>Aucun bilan consulté pour le moment.</p>
              )}

              <button
                className="rv-secondary"
                onClick={resetFeedback}
              >
                Réinitialiser les retours de test
              </button>
            </section>
          </>
        ) : screen === "devices" ? (
          <>
            <section className="rv-page-heading">
              <h1>Appareils et données</h1>
              <p>
                Choisissez comment alimenter
                les séances de vos sportifs.
              </p>
            </section>

            <div className="rv-device-grid">
              {([
                [FileUp, "Importation CSV",
                  "Disponible", "csv"],
                [Watch, "Montres connectées",
                  "Connexion à intégrer", "watch"],
                [Bluetooth, "Capteurs sportifs",
                  "Connexion à intégrer", "sensor"],
              ] as const).map(([Icon, label, status, id]) => (
                <section className="rv-panel" key={id}>
                  <Icon size={30} />
                  <h2>{label}</h2>
                  <p>{status}</p>

                  {id === "csv" ? (
                    <button
                      className="rv-primary"
                      onClick={() => navigate("athletes")}
                    >
                      Choisir un sportif
                    </button>
                  ) : (
                    <p className="rv-muted">
                      Pour le moment, importez
                      un fichier CSV compatible.
                    </p>
                  )}
                </section>
              ))}
            </div>

            <section className="rv-panel">
              <h2>Confidentialité</h2>

              <p>
                Les profils et retours de la maquette
                restent dans ce navigateur.
                Utilisez uniquement des données de test.
              </p>
            </section>
          </>
        ) : (
          <>
            <section className="rv-hero">
              <div>
                <small>VOTRE ESPACE COACH</small>

                <h1>Bienvenue sur RHYVA</h1>

                <p>
                  Comprenez mieux vos sportifs,
                  suivez leurs entraînements et gardez
                  la maîtrise de vos décisions.
                </p>

                <button
                  className="rv-primary"
                  onClick={() => navigate("athletes")}
                >
                  Mes sportifs <ArrowRight size={18} />
                </button>
              </div>

              <Activity size={100} strokeWidth={1} />
            </section>

            <div className="rv-metrics">
              {([
                [Users, "Sportifs suivis", athletes.length],
                [CalendarDays, "Séances", sessions.length],
                [AlertCircle, "Sportifs à examiner", prioritySessions],
              ] as const).map(([Icon, label, value]) => (
                <section className="rv-panel" key={label}>
                  <Icon size={24} />
                  <small>{label}</small>
                  <strong className="rv-big-number">
                    {value}
                  </strong>
                </section>
              ))}
            </div>

            <section className="rv-panel">
              <div className="rv-section-heading">
                <div>
                  <h2>Votre attention est recommandée</h2>
                  <p>
                    Uniquement les sportifs nécessitant
                    une revue prioritaire.
                  </p>
                </div>

                <span className="rv-tag rv-review">
                  {priorityGroups.length} sportif(s)
                </span>
              </div>

              {priorityGroups.length === 0 && (
                <div className="rv-callout rv-accept">
                  <CheckCircle2 size={25} />

                  <div>
                    <strong>
                      Aucune revue prioritaire en attente
                    </strong>
                    <p>
                      Selon les critères disponibles.
                    </p>
                  </div>
                </div>
              )}

              {priorityGroups.map((group) => (
                <div
                  className="rv-priority-row"
                  key={group.athlete.id}
                >
                  <span className="rv-avatar">
                    {group.athlete.initials}
                  </span>

                  <div className="rv-list-info">
                    <strong>{group.athlete.name}</strong>
                    <small>{group.athlete.sport}</small>
                    <p>{group.report.conclusion}</p>

                    <small>
                      {group.pendingCount}
                      {" "}séance(s) à examiner
                    </small>
                  </div>

                  <button
                    className="rv-primary"
                    onClick={() =>
                      openSession(group.target)
                    }
                  >
                    Examiner
                  </button>
                </div>
              ))}

              <p className="rv-muted">
                {monitoringCount} autre(s) sportif(s)
                en suivi conseillé, sans revue prioritaire.
              </p>
            </section>

            <section className="rv-panel">
              <div className="rv-section-heading">
                <div>
                  <h2>Mon groupe d'entraînement</h2>
                  <p>Retrouvez vos sportifs.</p>
                </div>

                <button
                  className="rv-primary"
                  onClick={() => navigate("athletes")}
                >
                  <Plus size={18} />
                  Gérer les sportifs
                </button>
              </div>

              {athletes.map((athlete) => {
                const last =
                  athlete.sessions[
                    athlete.sessions.length - 1
                  ];

                const status = last
                  ? getSessionStatus(last)
                  : {
                      label: "Aucune séance",
                      style: "neutral",
                    };

                return (
                  <button
                    key={athlete.id}
                    className="rv-list-row"
                    onClick={() => openAthlete(athlete.id)}
                  >
                    <span className="rv-avatar">
                      {athlete.initials}
                    </span>

                    <span className="rv-list-info">
                      <strong>{athlete.name}</strong>

                      <small>
                        {athlete.sport}
                        {" · "}
                        {athlete.sessions.length} séance(s)
                      </small>
                    </span>

                    <span className={
                      `rv-tag rv-${status.style}`
                    }>
                      {status.label}
                    </span>

                    <ArrowRight size={18} />
                  </button>
                );
              })}
            </section>
          </>
        )}

        <footer className="rv-footer">
          RHYVA · Aide au suivi sportif ·
          Pas de diagnostic médical
        </footer>
      </main>
    </div>
  );
}

export default App;
