
export type AthleteStatus = "review" | "normal" | "new";

export interface Athlete {
  id: string;
  name: string;
  initials: string;
  sport: string;
  duration: number;
  heartRate: number;
  personalRange: string;
  status: AthleteStatus;
  statusLabel: string;
  observation: string;
  explanation: string;
  heartRateCurve: number[];
}

export const athletes: Athlete[] = [
  {
    id: "maya",
    name: "Maya Benali",
    initials: "MB",
    sport: "Course à pied",
    duration: 58,
    heartRate: 118,
    personalRange: "96–108 bpm",
    status: "review",
    statusLabel: "À examiner",
    observation: "Une récupération différente de l'habitude",
    explanation:
      "Huit minutes après l'effort, la fréquence cardiaque reste au-dessus du repère individuel de cet exemple. Le coach peut examiner le contexte de la séance, sans conclure à une cause particulière.",
    heartRateCurve: [
      82, 86, 95, 117, 136, 152, 164, 170,
      158, 172, 166, 176, 165, 168, 161,
      149, 143, 135, 128, 118
    ]
  },
  {
    id: "lucas",
    name: "Lucas Martin",
    initials: "LM",
    sport: "Cyclisme",
    duration: 72,
    heartRate: 94,
    personalRange: "89–104 bpm",
    status: "normal",
    statusLabel: "Suivi habituel",
    observation: "Une récupération proche du profil habituel",
    explanation:
      "Dans cet exemple, la mesure observée après l'effort se situe dans la plage de comparaison individuelle. Le contexte reste nécessaire pour interpréter cette observation.",
    heartRateCurve: [
      75, 90, 110, 125, 134, 148, 139,
      162, 151, 156, 167, 156, 148,
      139, 127, 108, 104, 99, 94
    ]
  },
  {
    id: "ines",
    name: "Inès Rousseau",
    initials: "IR",
    sport: "Course à pied",
    duration: 45,
    heartRate: 112,
    personalRange: "99–115 bpm",
    status: "new",
    statusLabel: "À consulter",
    observation: "Une nouvelle séance à interpréter",
    explanation:
      "Les mesures fictives de cette séance ne permettent pas de conclure à un problème. Rhyva présente les informations pour faciliter la lecture du coach.",
    heartRateCurve: [
      79, 90, 110, 140, 154, 165, 161,
      155, 169, 163, 158, 170, 165,
      156, 149, 142, 131, 123, 116, 112
    ]
  }
];
