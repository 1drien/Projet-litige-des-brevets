# Modèle de Prédiction des Litiges sur les Brevets

Ce projet vise à construire un modèle de machine learning capable d’anticiper le risque de litige associé à un brevet au moment de son dépôt. Dans un contexte d’innovation technologique rapide et d’interdépendance croissante entre inventions, cette tâche s’avère cruciale, notamment pour les PME ne disposant pas de moyens juridiques avancés.

---

## Objectifs

- Développer un modèle prédictif robuste pour la détection précoce des litiges brevets.
- Comparer des approches linéaires et non linéaires (régression, forêts, boosting, réseaux de neurones).
- Améliorer la sensibilité du modèle à la classe minoritaire via des techniques de rééquilibrage.
- Proposer un outil interprétable pour les acteurs de la propriété intellectuelle.

---

## Données

Le jeu de données regroupe plusieurs milliers de brevets, caractérisés par :

- Informations temporelles : dates de dépôt, durée d’examen
- Données géographiques : pays d’origine, priorité étrangère
- Indicateurs de qualité : indices de diversité, de généralité, nombre de citations (avant/après)
- Contenu : nombre de revendications, statut universitaire, domaine technologique
- Cible : `Infringment` (binaire = 1 si litige, 0 sinon)

---

## Structure du Projet

```bash
.
├── data/                  # Données brutes et enrichies
├── src/                   # Scripts d'entraînement et traitement
│   ├── logistic_model.py
│   ├── xgboost_model.py
├── report/                # Rapport LaTeX final
├── requirements.txt       # Bibliothèques utilisées
└── README.md              # Présentation du projet

```
