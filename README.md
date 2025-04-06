<p align="center">
  <img src="https://img.icons8.com/ios-filled/100/000000/artificial-intelligence.png" alt="AI Logo" width="80"/>
</p>

<h1 align="center"> Modèle de Prédiction des Litiges sur les Brevets</h1>

<p align="center">
  Ce projet vise à construire un modèle de machine learning capable d’anticiper le risque de litige associé à un brevet au moment de son dépôt. Dans un contexte d’innovation technologique rapide et d’interdépendance croissante entre inventions, cette tâche s’avère cruciale, notamment pour les PME ne disposant pas de moyens juridiques avancés.
</p>

<p align="center">
  <img alt="GitHub repo stars" src="https://img.shields.io/github/stars/1drien/Projet-litige-des-brevets?style=social">
  <img alt="GitHub issues" src="https://img.shields.io/github/issues/1drien/Projet-litige-des-brevets">
  <img alt="GitHub license" src="https://img.shields.io/github/license/1drien/Projet-litige-des-brevets">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-en%20cours-yellow">
</p>

## Objectifs

- Développer un modèle prédictif robuste pour la détection précoce des litiges brevets.
- Comparer des approches linéaires et non linéaires.
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

## Langage utilisé

Le projet a été entièrement développé en **Python**, en raison de :

- son large écosystème pour le machine learning (`scikit-learn`, `tensorflow`, `xgboost`, etc.),
- sa syntaxe claire facilitant le prototypage rapide,
- et sa compatibilité avec des interfaces graphiques simples via `tkinter`.

Ce choix garantit une solution cohérente, maintenable et accessible.

---

## Structure du Projet

```bash
PROJET_MI/
├── interface/
│   ├── logs.txt                  # Sauvegarde des brevets prédits comme litigieux
│   ├── ProgramInterface.py       # Lancement de l’interface utilisateur
│   ├── UI_Testing_Version.py     # Version de test alternative
│   ├── scaler_fold_1.pkl         # Metriques du MLP pour le pourcentage
│   └── model_fold_1.h5           # Modèle MLP entraîné
│
├── models/
│   ├── reg.py                    # Modèle de régression logistique
│   ├── ElasticNet.py             # Modèle Elastic Net
│   ├── baysar.py                 # Random Forest
│   ├── xgboost_model.py          # Modèle XGBoost
│   ├── NeuralNetwork_Modified.py # Réseau de neurones (MLP)
│   ├── Dataset_Thuy (1).csv      # Données complètes
│   └── Dataset.csv               # Dataset pour le réseau de neurones
│
├── requirements.txt              # Dépendances Python
├── README.md                     # Présentation du projet
├── .gitignore                    # Fichiers ignorés par Git
└── assets/                       # Images / captures d'écran pour le README
```

---

## Exécution des modèles

Assurez-vous d’avoir installé les dépendances nécessaires (voir plus bas ⬇️).

Chaque script peut être exécuté indépendamment pour entraîner et tester un modèle :

| Modèle                | Script à exécuter                  |
| --------------------- | ---------------------------------- |
| Régression logistique | `models/reg.py`                    |
| Elastic Net           | `models/ElasticNet.py`             |
| XGBoost               | `models/xgboost_model.py`          |
| Réseau de neurones    | `models/NeuralNetwork_Modified.py` |
| Random Forest         | `models/baysar.py`                 |

### Exemple : exécuter le modèle XGBoost

```bash
python models/xgboost_model.py
```

Les résultats s’affichent directement dans la console (matrice de confusion, F1-score, ratio TP/FP, etc.).

---

## Interface utilisateur

Une interface graphique permet de charger les caractéristiques d’un brevet et de prédire son risque de litige.

### Lancer l’interface :

```bash
python interface/ProgramInterface.py
```

Une fenêtre s’ouvrira pour permettre à l’utilisateur de saisir les informations d’un brevet ou de charger un fichier d’entrée.

> L’interface utilise le Réseau de neurones, qui est sauvegardé sous `interface/model_fold_1.h5`.

---

## Données

Le fichier de données se trouve ici :  
`models/Dataset_Thuy (1).csv`

---

## Dépendances et installation

Créez un environnement virtuel Python (optionnel mais recommandé) :

```bash
python -m venv venv
source venv/bin/activate   # Sous Windows : venv\Scripts\activate
```

Installez les dépendances avec :

```bash
pip install -r requirements.txt
```

### Contenu du fichier `requirements.txt` :

```txt
scikit-learn>=1.2.2
imblearn>=0.0
xgboost>=1.7.6
tensorflow>=2.11.0
numpy>=1.23.5
pandas>=1.5.3
matplotlib>=3.7.1
tk
joblib>=1.2.0

```

## Aperçu de l'interface

Voici un exemple de l'interface graphique permettant de prédire le risque de litige d'un brevet :

<p align="center">
  <img src="assets/infrigement.png" alt="Aperçu Interface" width="600"/>
</p>
