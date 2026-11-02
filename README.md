# Projet TER — Explainable Athlete Monitoring

Pipeline expérimental de reconnaissance d'états fonctionnels à partir de signaux cardio-respiratoires issus de SportDB 2.0.

Le système reconnaît trois états :

- Rest
- Effort
- Recovery

et fournit également :

- une confiance calibrée ;
- une explication locale SHAP ;
- un indicateur de cohérence sémantique.

Le projet constitue une base de recherche pour un futur système de suivi explicable de sportifs.

Il ne réalise pas de diagnostic médical.

## Baseline actuelle

La baseline stable utilise :

- HR + BR ;
- fenêtres de 50 secondes ;
- pas de 25 secondes ;
- calibration individuelle à partir d'une phase Rest ;
- 14 features personnalisées ;
- Random Forest ;
- validation Leave-One-Subject-Out ;
- calibration nested de la confiance selon l'état prédit ;
- TreeSHAP ;
- cohérence sémantique.

Résultats principaux sur 9 166 fenêtres d'évaluation :

| Métrique | Valeur |
|---|---:|
| Accuracy | 0.7551 |
| Balanced accuracy | 0.6615 |
| Macro-F1 | 0.6310 |
| Recall Rest | 0.6286 |
| Recall Effort | 0.8211 |
| Recall Recovery | 0.5348 |

Après calibration de la confiance :

| Métrique | Valeur |
|---|---:|
| Brier | 0.1319 |
| ECE | 0.0119 |
| ROC-AUC de fiabilité | 0.8265 |

Les décisions scientifiques détaillées sont documentées dans `BASELINE_STABLE.md`.

## Structure

```text
src/core/
```

Contient les composants scientifiques de la baseline :

- chargement du dataset ;
- préparation des features ;
- modèle ;
- calibration de confiance ;
- explicabilité et cohérence sémantique.

```text
src/analysis/
```

Contient les scripts de contrôle qualité et d'analyse de la baseline.

```text
src/experiments/
```

Contient les expériences qui ne font pas partie du pipeline principal, notamment :

- calibration sport × état ;
- normalisation de la cohérence sémantique ;
- expériences HITL ;
- expériences spécifiques à Recovery.

```text
legacy/
```

Contient les scripts et résultats issus des versions précédentes du projet.

```text
results/
```

Contient les résultats générés par les expériences actuelles.

## Dataset

Le projet utilise SportDB 2.0.

Le dataset n'est pas versionné dans le dépôt Git.

L'arborescence attendue est :

```text
Projet_TER/
└── SportDB2/
```

Le corpus principal actuel utilise :

- AMF ;
- CYC ;
- RUN ;
- SOC.

## Exécution

Depuis la racine du projet, utiliser les modules Python.

Exemple :

```bash
python3 -m src.analysis.data_quality
```

Expérience du modèle principal :

```bash
python3 -m src.core.modeling
```

Explicabilité :

```bash
python3 -m src.core.semantic
```

Calibration nested :

```bash
python3 -m src.core.calibration
```

Diagnostics :

```bash
python3 -m src.analysis.model_diagnostics
```

## Résultats générés

Les sorties de la baseline sont enregistrées dans :

```text
results/baseline/
```

Les sorties des expériences secondaires sont enregistrées dans :

```text
results/experiments/
```

Ces résultats sont générés à partir du code et ne constituent pas le code source du projet.

## HITL

Les expériences Accept / Check / Review ont montré une capacité intéressante à concentrer les erreurs dans les cas à examiner.

Cependant, leur robustesse entre sujets est insuffisante pour considérer actuellement la catégorie `Accept` comme une garantie automatique de fiabilité.

Le HITL reste donc un module expérimental de priorisation humaine.

## Limites

Les principales limites actuelles sont :

- forte variabilité inter-sujets ;
- difficulté particulière de Recovery ;
- déséquilibre en faveur d'Effort ;
- performances plus faibles sur AMF ;
- un seul vrai sujet SOC ;
- nécessité d'une phase Rest pour la calibration individuelle ;
- absence de labels de blessure ;
- absence de validation clinique.

## Prochaine phase

La prochaine étape consiste à construire un moteur applicatif stable autour de la baseline :

```text
HR + BR
    ↓
calibration personnelle
    ↓
features
    ↓
classification
    ↓
confiance calibrée
    ↓
SHAP
    ↓
cohérence sémantique
    ↓
sortie applicative
```

Ce moteur servira ensuite de base à une API et à une future interface de suivi.
