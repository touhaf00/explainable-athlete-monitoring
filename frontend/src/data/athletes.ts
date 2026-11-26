export type AthleteStatus = "review" | "normal" | "new";

export interface PhysiologicalSample {
  minute: number;
  heartRate: number;
  respiratoryRate: number;
}

export interface ReferenceSample {
  minute: number;
  heartRate: number;
}

export interface Athlete {
  id: string;
  name: string;
  initials: string;
  sport: string;
  sessionType: string;
  duration: number;
  heartRate: number;
  personalRange: string;
  status: AthleteStatus;
  statusLabel: string;
  observation: string;
  explanation: string;
  restEnd: number;
  effortEnd: number;
  classificationConfidence: number;
  samples: PhysiologicalSample[];
  recoveryReference: ReferenceSample[];
  heartRateCurve: number[];
}

const scenarios: Omit<Athlete, "heartRateCurve">[] = [
  {
    id: "maya",
    name: "Maya Benali",
    initials: "MB",
    sport: "Course à pied",
    sessionType: "Entraînement fractionné",
    duration: 58,
    heartRate: 118,
    personalRange: "96–108 bpm",
    status: "review",
    statusLabel: "À examiner",
    restEnd: 6,
    effortEnd: 50,
    classificationConfidence: 74,

    observation:
      "Une récupération plus lente que son repère habituel",

    explanation:
      "Dans cette simulation, la fréquence cardiaque reste supérieure au repère personnel huit minutes après un effort fractionné. Le coach peut examiner la charge de la séance et le ressenti du sportif, sans conclure à une cause particulière.",

    samples: [
      { minute: 0, heartRate: 72, respiratoryRate: 13 },
      { minute: 3, heartRate: 75, respiratoryRate: 14 },
      { minute: 6, heartRate: 80, respiratoryRate: 16 },
      { minute: 10, heartRate: 165, respiratoryRate: 32 },
      { minute: 14, heartRate: 130, respiratoryRate: 25 },
      { minute: 18, heartRate: 174, respiratoryRate: 36 },
      { minute: 22, heartRate: 138, respiratoryRate: 27 },
      { minute: 26, heartRate: 180, respiratoryRate: 39 },
      { minute: 30, heartRate: 142, respiratoryRate: 29 },
      { minute: 34, heartRate: 182, respiratoryRate: 40 },
      { minute: 38, heartRate: 145, respiratoryRate: 30 },
      { minute: 42, heartRate: 177, respiratoryRate: 38 },
      { minute: 46, heartRate: 150, respiratoryRate: 32 },
      { minute: 50, heartRate: 172, respiratoryRate: 37 },
      { minute: 52, heartRate: 150, respiratoryRate: 30 },
      { minute: 54, heartRate: 136, respiratoryRate: 25 },
      { minute: 56, heartRate: 125, respiratoryRate: 22 },
      { minute: 58, heartRate: 118, respiratoryRate: 20 }
    ],

    recoveryReference: [
      { minute: 50, heartRate: 172 },
      { minute: 52, heartRate: 135 },
      { minute: 54, heartRate: 119 },
      { minute: 56, heartRate: 109 },
      { minute: 58, heartRate: 102 }
    ]
  },

  {
    id: "lucas",
    name: "Lucas Martin",
    initials: "LM",
    sport: "Cyclisme",
    sessionType: "Endurance continue",
    duration: 72,
    heartRate: 94,
    personalRange: "89–104 bpm",
    status: "normal",
    statusLabel: "Suivi habituel",
    restEnd: 8,
    effortEnd: 64,
    classificationConfidence: 93,

    observation:
      "Une récupération proche du profil habituel",

    explanation:
      "Dans cet exemple fictif, la fréquence cardiaque reste relativement stable pendant l'effort et rejoint progressivement le repère individuel pendant la récupération. Le contexte de la séance reste nécessaire à son interprétation.",

    samples: [
      { minute: 0, heartRate: 65, respiratoryRate: 12 },
      { minute: 4, heartRate: 67, respiratoryRate: 13 },
      { minute: 8, heartRate: 78, respiratoryRate: 15 },
      { minute: 12, heartRate: 110, respiratoryRate: 21 },
      { minute: 16, heartRate: 135, respiratoryRate: 26 },
      { minute: 20, heartRate: 143, respiratoryRate: 28 },
      { minute: 26, heartRate: 145, respiratoryRate: 29 },
      { minute: 32, heartRate: 140, respiratoryRate: 27 },
      { minute: 38, heartRate: 148, respiratoryRate: 30 },
      { minute: 44, heartRate: 144, respiratoryRate: 28 },
      { minute: 50, heartRate: 147, respiratoryRate: 29 },
      { minute: 56, heartRate: 142, respiratoryRate: 27 },
      { minute: 60, heartRate: 145, respiratoryRate: 28 },
      { minute: 64, heartRate: 140, respiratoryRate: 27 },
      { minute: 66, heartRate: 115, respiratoryRate: 22 },
      { minute: 68, heartRate: 104, respiratoryRate: 18 },
      { minute: 70, heartRate: 98, respiratoryRate: 15 },
      { minute: 72, heartRate: 94, respiratoryRate: 14 }
    ],

    recoveryReference: [
      { minute: 64, heartRate: 140 },
      { minute: 66, heartRate: 118 },
      { minute: 68, heartRate: 107 },
      { minute: 70, heartRate: 101 },
      { minute: 72, heartRate: 97 }
    ]
  },

  {
    id: "ines",
    name: "Inès Rousseau",
    initials: "IR",
    sport: "Course à pied",
    sessionType: "Effort progressif",
    duration: 45,
    heartRate: 112,
    personalRange: "99–115 bpm",
    status: "new",
    statusLabel: "À consulter",
    restEnd: 5,
    effortEnd: 37,
    classificationConfidence: 68,

    observation:
      "Une séance progressive à interpréter",

    explanation:
      "Dans cette simulation, l'intensité augmente progressivement jusqu'à la fin de l'effort. La dernière mesure se situe dans le repère personnel, mais la classification illustrative présente une incertitude plus importante. Le coach peut compléter l'analyse avec ses observations.",

    samples: [
      { minute: 0, heartRate: 74, respiratoryRate: 14 },
      { minute: 2, heartRate: 76, respiratoryRate: 14 },
      { minute: 5, heartRate: 85, respiratoryRate: 16 },
      { minute: 8, heartRate: 100, respiratoryRate: 19 },
      { minute: 11, heartRate: 114, respiratoryRate: 22 },
      { minute: 14, heartRate: 125, respiratoryRate: 24 },
      { minute: 17, heartRate: 132, respiratoryRate: 26 },
      { minute: 20, heartRate: 140, respiratoryRate: 28 },
      { minute: 23, heartRate: 148, respiratoryRate: 30 },
      { minute: 26, heartRate: 154, respiratoryRate: 32 },
      { minute: 29, heartRate: 162, respiratoryRate: 34 },
      { minute: 32, heartRate: 170, respiratoryRate: 37 },
      { minute: 35, heartRate: 177, respiratoryRate: 39 },
      { minute: 37, heartRate: 183, respiratoryRate: 42 },
      { minute: 39, heartRate: 156, respiratoryRate: 33 },
      { minute: 41, heartRate: 137, respiratoryRate: 27 },
      { minute: 43, heartRate: 121, respiratoryRate: 22 },
      { minute: 45, heartRate: 112, respiratoryRate: 19 }
    ],

    recoveryReference: [
      { minute: 37, heartRate: 183 },
      { minute: 39, heartRate: 150 },
      { minute: 41, heartRate: 130 },
      { minute: 43, heartRate: 116 },
      { minute: 45, heartRate: 107 }
    ]
  }
];

export const athletes: Athlete[] = scenarios.map(
  (athlete) => ({
    ...athlete,
    heartRateCurve: athlete.samples.map(
      (sample) => sample.heartRate
    )
  })
);
