# Baseline stable — Projet TER / Athlete Monitoring

## 1. Statut

Cette version constitue la baseline scientifique stable du projet.

À partir de ce point, le corpus, le protocole principal, le modèle de référence et la méthode de calibration ne doivent plus être modifiés sans raison scientifique explicite ou découverte d'une erreur dans le pipeline.

L'objectif est maintenant d'utiliser cette base pour construire le moteur applicatif du projet.

---

## 2. Objectif du système

Le système vise à reconnaître trois états fonctionnels à partir de signaux physiologiques :

- Rest
- Effort
- Recovery

Le pipeline doit également fournir :

- une confiance calibrée ;
- une explication locale par SHAP ;
- une cohérence sémantique de l'explication ;
- une indication permettant d'identifier les prédictions qui nécessitent davantage de prudence.

Le système n'est pas destiné à établir un diagnostic médical ni à prédire directement une blessure.

---

## 3. Dataset

Source principale :

`SportDB 2.0`

Corpus principal retenu :

- AMF
- CYC
- RUN
- SOC

Signaux utilisés :

- HR
- BR

RR a été étudié séparément dans une ablation contrôlée mais n'est pas retenu dans le modèle principal.

### Corpus principal

- 42 enregistrements
- 38 vrais sujets
- 9 628 fenêtres avant exclusion calibration/overlap
- 420 fenêtres utilisées pour la calibration individuelle
- 42 fenêtres supplémentaires exclues car elles chevauchent la zone utilisée pour la calibration
- 9 166 fenêtres finales d'évaluation

SOC est regroupé sous un seul vrai sujet :

`SOC/SOC3/S1`

Les différentes sessions CRD de ce sujet ne sont pas considérées comme des individus indépendants.

---

## 4. Fenêtrage

Paramètres retenus :

- fréquence : 1 Hz
- taille de fenêtre : 50 secondes
- pas : 25 secondes

Une fenêtre utilisée pour l'évaluation ne doit pas chevaucher la zone servant à construire la calibration individuelle.

---

## 5. Calibration individuelle

Pour chaque enregistrement, les 10 premières fenêtres Rest sont utilisées afin d'établir une baseline personnelle.

La baseline utilise notamment les médianes HR et BR.

Les fenêtres servant à cette calibration ne sont pas utilisées pour l'évaluation.

Cette calibration suppose qu'en situation réelle une courte phase de repos connue est disponible avant l'analyse.

---

## 6. Représentation principale

Feature set retenu :

`personalized_14`

Il contient 14 variables.

### Statistiques HR/BR

6 variables :

- mean HR
- std HR
- max HR
- mean BR
- std BR
- max BR

### Variables temporelles

4 variables :

- slope HR
- slope BR
- delta HR
- delta BR

### Variables personnalisées

4 variables :

- mean HR delta par rapport à la calibration
- mean BR delta par rapport à la calibration
- mean HR ratio par rapport à la calibration
- mean BR ratio par rapport à la calibration

---

## 7. Modèle principal

Modèle :

`RandomForestClassifier`

Configuration principale :

- 300 arbres
- `max_depth = 8`
- `min_samples_leaf = 3`
- `class_weight = balanced`
- `random_state = 42`

Le Random Forest reste le modèle de référence du socle actuel.

Un benchmark supplémentaire pourra être réalisé ultérieurement comme contrôle, mais il ne fait pas partie des prérequis nécessaires au développement applicatif.

---

## 8. Protocole d'évaluation

Le protocole principal est :

`Leave-One-Subject-Out`

Chaque sujet est laissé complètement hors entraînement à son tour.

Aucune fenêtre appartenant au sujet de test n'est utilisée pour entraîner le classifieur.

Le protocole produit donc des prédictions OOF sur les 38 sujets.

---

## 9. Résultats principaux

Sur les 9 166 fenêtres finales :

| Métrique | Résultat |
|---|---:|
| Accuracy | 0.7551 |
| Balanced accuracy | 0.6615 |
| Macro-F1 | 0.6310 |
| Recall Rest | 0.6286 |
| Recall Effort | 0.8211 |
| Recall Recovery | 0.5348 |

La principale difficulté reste l'état Recovery.

La confusion dominante pour Recovery est :

`Recovery → Effort`

Matrice de confusion :

| Vrai état | Rest | Effort | Recovery |
|---|---:|---:|---:|
| Rest | 672 | 198 | 199 |
| Effort | 442 | 5503 | 757 |
| Recovery | 256 | 393 | 746 |

---

## 10. Performances par sport

| Sport | Accuracy | Balanced accuracy |
|---|---:|---:|
| AMF | 0.7050 | 0.6065 |
| CYC | 0.8548 | 0.7226 |
| RUN | 0.8736 | 0.8359 |
| SOC | 0.6761 | 0.7060 |

AMF présente une forte variabilité inter-sujets.

SOC ne contient qu'un seul vrai sujet dans le corpus utilisé. Les conclusions spécifiques à SOC doivent donc être considérées avec prudence.

---

## 11. Ablation RR

Une comparaison contrôlée a été réalisée uniquement sur les fenêtres disposant de RR de qualité suffisante.

Corpus apparié :

- 31 enregistrements
- 27 sujets
- 7 101 fenêtres finales d'évaluation

Résultats :

| Feature set | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| personalized_14 HR+BR | 0.7486 | 0.6515 | 0.6283 |
| personalized_22 HR+BR+RR | 0.7471 | 0.6469 | 0.6258 |

Conclusion :

L'ajout de RR ne produit pas d'amélioration observable dans ce protocole contrôlé.

RR n'est donc pas requis dans le modèle principal actuel.

---

## 12. Explicabilité

TreeSHAP est utilisé pour produire une explication locale de chaque prédiction OOF.

SHAP est calculé dans chaque fold LOSO sur le modèle correspondant.

L'explication relie les variables HR et BR aux concepts physiologiques utilisés dans la couche sémantique.

---

## 13. Cohérence sémantique

Pour les variables SHAP positives les plus importantes, une similarité avec les profils physiologiques des états est calculée.

Jusqu'à 5 variables positives sont utilisées.

La cohérence sémantique est définie par :

\[
C_i =
\frac{
\sum_{j \in P_i}
|\phi_{ij}|
q_{ij\hat y_i}
}{
\sum_{j \in P_i}
|\phi_{ij}|
}
\]

Les profils utilisés pour évaluer le sujet test sont construits uniquement à partir des données d'entraînement.

Si le sport du sujet de test n'existe pas dans les données d'entraînement, les profils globaux sont utilisés.

La cohérence sémantique est conservée comme indicateur explicatif distinct.

---

## 14. Analyse confiance / cohérence / fiabilité

Avant calibration :

| Score | ROC-AUC |
|---|---:|
| Confiance brute | 0.7237 |
| Cohérence | 0.7164 |
| sqrt(confiance × cohérence) | 0.7528 |

Le score combiné améliorait donc initialement le classement des prédictions correctes.

Cependant, cette conclusion change une fois la confiance correctement calibrée.

---

## 15. Calibration de la confiance

La confiance brute du Random Forest présente des biais importants selon l'état prédit.

Une calibration nested par état prédit a donc été mise en place.

Elle est apprise uniquement à partir des sujets d'entraînement.

Résultats globaux :

| Métrique | Brut | Calibré |
|---|---:|---:|
| Mean confidence | 0.6869 | 0.7535 |
| Accuracy | 0.7551 | 0.7551 |
| Brier | 0.1686 | 0.1319 |
| ECE | 0.0682 | 0.0119 |
| ROC-AUC fiabilité | 0.7237 | 0.8265 |

La calibration nested par état est donc retenue.

La confiance utilisée dans la suite est :

\[
\tilde p
\]

---

## 16. Calibration sport × état

Une calibration supplémentaire conditionnée par sport × état a été testée.

Résultats :

- Brier : 0.1298
- ECE : 0.0366
- ROC-AUC : 0.8311

Elle améliore légèrement Brier et ROC-AUC mais détériore nettement l'ECE et présente une forte instabilité dans certains sous-groupes.

Elle n'est donc pas retenue dans la baseline.

La calibration définitive reste la calibration nested par état prédit.

---

## 17. Combinaison avec la cohérence sémantique

Après calibration :

\[
AUC(\tilde p) = 0.8265
\]

alors que :

\[
AUC(\sqrt{\tilde p C}) = 0.8117
\]

La multiplication de la confiance calibrée avec la cohérence n'améliore donc pas le classement des prédictions fiables.

Des normalisations de C par percentile d'état et sport × état ont également été testées et n'ont pas apporté d'amélioration suffisante.

Décision :

- la confiance calibrée est utilisée pour estimer la fiabilité prédictive ;
- la cohérence sémantique reste une information explicative séparée ;
- C n'est pas transformé en seconde probabilité et n'est pas forcé dans un score combiné.

---

## 18. HITL

Un premier système :

- Accept
- Check
- Review

a été évalué de manière nested.

Le système réussit à concentrer une partie importante des erreurs dans Review, mais la généralisation inter-sujets n'est pas suffisamment robuste.

Dans l'expérience robuste :

- Accept : 2 468 fenêtres
- couverture Accept : 26.93 %
- accuracy Accept : 93.72 %
- Check : 4 995 fenêtres
- Review : 1 703 fenêtres
- Review capture environ 49.31 % des erreurs

Cependant, le pire sujet disposant de prédictions Accept n'obtient que :

`54.69 %`

De plus, la catégorie Accept est très fortement dominée par Effort.

Décision :

Le mécanisme Accept / Check / Review n'est pas validé comme garantie automatique de fiabilité.

Il reste un module expérimental de priorisation humaine.

Il pourra être amélioré ultérieurement avec davantage de sujets et une meilleure représentation des contextes physiologiques.

---

## 19. Pipeline stable retenu

Le socle scientifique retenu est :

```text
SportDB2
    ↓
HR + BR
    ↓
contrôle qualité
    ↓
fenêtres 50 s / pas 25 s
    ↓
baseline personnelle Rest
    ↓
personalized_14
    ↓
Random Forest
    ↓
Rest / Effort / Recovery
    ↓
confiance nested calibrée par état
    ↓
TreeSHAP
    ↓
graphe / profils sémantiques
    ↓
cohérence C
    ↓
sortie explicable
```

---

## 20. Sortie prévue du moteur

À terme, une prédiction devra pouvoir retourner au minimum :

```text
predicted_state
calibrated_confidence
shap_explanation
semantic_consistency
caution / human-review information
```

La cohérence sémantique ne modifie pas automatiquement la probabilité calibrée.

---

## 21. Limites actuelles

Les principales limites identifiées sont :

- variabilité inter-sujets importante ;
- performances plus faibles sur AMF ;
- Recovery reste difficile à reconnaître ;
- dataset déséquilibré vers Effort ;
- SOC contient un seul vrai sujet ;
- calibration individuelle nécessitant une phase Rest connue ;
- absence de labels de blessure ;
- le HITL Accept/Check/Review n'est pas suffisamment robuste pour constituer une garantie automatique ;
- les résultats sont liés au corpus SportDB2 et ne constituent pas une validation clinique.

---

## 22. Éléments rejetés de la baseline

Ne sont pas retenus comme composants principaux :

- RR obligatoire ;
- score de fiabilité brut `sqrt(p × C)` ;
- score `sqrt(calibrated_p × C)` ;
- normalisation percentile de C ;
- calibration sport × état comme calibration principale ;
- seuils globaux Accept / Check / Review considérés comme garantie automatique.

Les scripts correspondants sont conservés comme traces expérimentales.

---

## 23. Règle de gel

À partir de cette baseline :

- aucun changement du corpus principal sans justification ;
- aucun changement de feature set sans nouvelle expérience clairement identifiée ;
- aucun tuning opportuniste à partir des résultats OOF ;
- aucun nouveau seuil HITL choisi après observation du sujet test ;
- aucune modification des métriques de référence sans nouvelle version explicitement documentée.

Les travaux suivants concernent prioritairement l'architecture applicative.

---

## 24. Prochaine phase

Prochaine phase du projet :

### Moteur de prédiction

Construire un composant réutilisable capable de recevoir :

- HR ;
- BR ;
- informations de baseline personnelle.

Et de produire :

- état Rest / Effort / Recovery ;
- confiance calibrée ;
- explication SHAP ;
- cohérence sémantique ;
- informations de prudence.

Puis :

```text
moteur Python
    ↓
API
    ↓
historique temporel
    ↓
application mobile / interface
```

---

## 25. Version de référence

Cette baseline devient la version de référence avant développement applicatif.

Date de gel :

`02/11/2026`

## 26. Reproductibilité numérique

La baseline a été exécutée plusieurs fois après refactorisation.

Les prédictions de classe, les labels de correction et les métriques agrégées sont identiques entre exécutions.

De très faibles variations numériques peuvent apparaître dans certaines valeurs flottantes de confiance et de fiabilité, avec un écart maximal inférieur à `1e-12`.

Ces variations sont considérées comme du bruit numérique et n'affectent aucune décision, prédiction ou conclusion expérimentale.
