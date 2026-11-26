import { useState } from "react";
import type { FormEvent } from "react";

import {
  UserPlus,
  Search,
  Pencil,
  Trash2,
  ArrowRight,
  X,
  Users,
} from "lucide-react";

import type { SportAthlete } from "../data/sessionModels";

import "./AthleteManager.css";

interface Props {
  athletes: SportAthlete[];
  onChange: (athletes: SportAthlete[]) => void;
  onOpen: (athleteId: string) => void;
}

const sports = [
  "Course à pied",
  "Cyclisme",
  "Triathlon",
  "Natation",
  "Football",
  "Football américain",
  "Basketball",
  "Tennis",
  "Musculation",
  "Autre",
];

const levels = [
  "Débutant",
  "Intermédiaire",
  "Confirmé",
  "Professionnel",
];

const emptyForm = {
  firstName: "",
  lastName: "",
  sport: "Course à pied",
  customSport: "",
  level: "Intermédiaire",
  objectives: "",
};

export default function AthleteManager({
  athletes,
  onChange,
  onOpen,
}: Props) {
  const [search, setSearch] = useState("");
  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] =
    useState<string | null>(null);

  const [form, setForm] = useState(emptyForm);

  const filtered = athletes.filter((athlete) =>
    `${athlete.name} ${athlete.sport}`
      .toLowerCase()
      .includes(search.trim().toLowerCase())
  );

  function startCreation() {
    setEditingId(null);
    setForm({ ...emptyForm });
    setFormOpen(true);
  }

  function startEditing(athlete: SportAthlete) {

    const [firstName, ...lastName] =
      athlete.name.split(" ");

    setEditingId(athlete.id);

    setForm({
      firstName,
      lastName: lastName.join(" "),
      sport: sports.includes(athlete.sport)
        ? athlete.sport
        : "Autre",
      customSport: sports.includes(athlete.sport)
        ? ""
        : athlete.sport,
      level: athlete.level ?? "Intermédiaire",
      objectives: athlete.objectives ?? "",
    });

    setFormOpen(true);
  }

  function closeForm() {
    setFormOpen(false);
    setEditingId(null);
    setForm({ ...emptyForm });
  }

  function saveAthlete(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const firstName = form.firstName.trim();
    const lastName = form.lastName.trim();

    const sport =
      form.sport === "Autre"
        ? form.customSport.trim()
        : form.sport;

    if (!firstName || !lastName || !sport) return;

    const existing = athletes.find(
      (athlete) => athlete.id === editingId
    );

    if (editingId && !existing) return;

    const athlete: SportAthlete = {
      id: existing?.id ?? `coach-${crypto.randomUUID()}`,
      name: `${firstName} ${lastName}`,
      initials:
        `${firstName[0]}${lastName[0]}`.toUpperCase(),
      sport,
      level: form.level,
      objectives: form.objectives.trim(),
      source: existing?.source ?? "coach",
      fictionalIdentity: existing?.fictionalIdentity ?? true,
      sessions: existing?.sessions ?? [],
    };

    onChange(
      existing
        ? athletes.map((item) =>
            item.id === existing.id ? athlete : item
          )
        : [...athletes, athlete]
    );

    closeForm();
  }

  function deleteAthlete(athlete: SportAthlete) {

    const confirmed = window.confirm(
      `Supprimer ${athlete.name} et toutes ses séances ?`
    );

    if (!confirmed) return;

    onChange(
      athletes.filter((item) => item.id !== athlete.id)
    );

    if (editingId === athlete.id) closeForm();
  }

  return (
    <section className="athlete-manager">
      <div className="am-header">
        <div>
          <h1>Mes sportifs</h1>
          <p>
            Gérez vos sportifs et suivez leurs séances.
          </p>
        </div>

        <button
          type="button"
          className="am-primary"
          onClick={startCreation}
        >
          <UserPlus size={18} />
          Ajouter un sportif
        </button>
      </div>

      <div className="am-stats">
        <div>
          <span>Sportifs</span>
          <strong>{athletes.length}</strong>
        </div>

        <div>
          <span>Sportifs ajoutés</span>
          <strong>
            {athletes.filter(
              (a) => a.source === "coach"
            ).length}
          </strong>
        </div>

        <div>
          <span>Séances enregistrées</span>
          <strong>
            {athletes.reduce(
              (total, a) => total + a.sessions.length,
              0
            )}
          </strong>
        </div>
      </div>

      {formOpen && (
        <form
          className="am-form"
          onSubmit={saveAthlete}
        >
          <div className="am-header">
            <h2>
              {editingId
                ? "Modifier le sportif"
                : "Nouveau sportif"}
            </h2>

            <button
              type="button"
              className="am-icon-button"
              aria-label="Fermer le formulaire"
              onClick={closeForm}
            >
              <X size={20} />
            </button>
          </div>

          <p className="am-notice">
            Version de démonstration : utilisez une
            identité fictive et des données de test.
          </p>

          <div className="am-form-grid">
            <label>
              Prénom
              <input
                required
                maxLength={60}
                value={form.firstName}
                onChange={(e) =>
                  setForm({
                    ...form,
                    firstName: e.target.value,
                  })
                }
                placeholder="Prénom"
              />
            </label>

            <label>
              Nom
              <input
                required
                maxLength={60}
                value={form.lastName}
                onChange={(e) =>
                  setForm({
                    ...form,
                    lastName: e.target.value,
                  })
                }
                placeholder="Nom"
              />
            </label>

            <label>
              Discipline
              <select
                value={form.sport}
                onChange={(e) =>
                  setForm({
                    ...form,
                    sport: e.target.value,
                  })
                }
              >
                {sports.map((sport) => (
                  <option key={sport}>{sport}</option>
                ))}
              </select>
            </label>

            <label>
              Niveau
              <select
                value={form.level}
                onChange={(e) =>
                  setForm({
                    ...form,
                    level: e.target.value,
                  })
                }
              >
                {levels.map((level) => (
                  <option key={level}>{level}</option>
                ))}
              </select>
            </label>

            {form.sport === "Autre" && (
              <label className="am-full">
                Préciser la discipline
                <input
                  required
                  maxLength={80}
                  value={form.customSport}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      customSport: e.target.value,
                    })
                  }
                />
              </label>
            )}

            <label className="am-full">
              Objectifs sportifs
              <textarea
                rows={3}
                maxLength={1000}
                value={form.objectives}
                onChange={(e) =>
                  setForm({
                    ...form,
                    objectives: e.target.value,
                  })
                }
                placeholder="Objectifs, préparation..."
              />
            </label>
          </div>

          <div className="am-actions">
            <button
              type="button"
              className="am-secondary"
              onClick={closeForm}
            >
              Annuler
            </button>

            <button
              type="submit"
              className="am-primary"
            >
              {editingId
                ? "Enregistrer"
                : "Créer le sportif"}
            </button>
          </div>
        </form>
      )}

      <div className="am-search">
        <Search size={18} />
        <input
          aria-label="Rechercher un sportif"
          placeholder="Rechercher un sportif..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="am-list">
        {filtered.length === 0 && (
          <div className="am-empty">
            <Users size={32} />
            <p>Aucun sportif trouvé.</p>
          </div>
        )}

        {filtered.map((athlete) => (
          <article
            className="am-athlete"
            key={athlete.id}
          >
            <div className="am-avatar">
              {athlete.initials}
            </div>

            <div className="am-athlete-info">
              <h3>{athlete.name}</h3>

              <p>
                {athlete.sport}
                {athlete.level
                  ? ` · ${athlete.level}`
                  : ""}
              </p>

              <small>
                {athlete.sessions.length} séance(s)
                {" · "}
                {athlete.source === "sportdb2"
                  ? "Profil exemple"
                  : "Profil ajouté"}
              </small>
            </div>

            <div className="am-athlete-actions">
              {(
                <>
                  <button
                    type="button"
                    className="am-icon-button"
                    aria-label={`Modifier ${athlete.name}`}
                    onClick={() => startEditing(athlete)}
                  >
                    <Pencil size={17} />
                  </button>

                  <button
                    type="button"
                    className="am-icon-button am-delete"
                    aria-label={`Supprimer ${athlete.name}`}
                    onClick={() => deleteAthlete(athlete)}
                  >
                    <Trash2 size={17} />
                  </button>
                </>
              )}

              <button
                type="button"
                className="am-secondary"
                onClick={() => onOpen(athlete.id)}
              >
                Voir le profil
                <ArrowRight size={16} />
              </button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
