import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# ----------- Chargement et prétraitement ----------- #
file_path = 'Dataset_Thuy (1).csv'
df = pd.read_csv(file_path)

y = df["Infringment"].astype(int)
X = df.drop(columns=["Infringment", "pub_nbr", "Year_Litigation"])

# Imputation des NaN par médiane
X.fillna(X.median(numeric_only=True), inplace=True)

# Encodage fréquentiel des colonnes catégorielles
for col in X.select_dtypes(include=['object', 'category']).columns:
    freq = X[col].value_counts(normalize=True)
    X[col] = X[col].map(freq)

X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)

# ----------- Split train/test ----------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ----------- SMOTE après split ----------- #
smote = SMOTE(random_state=42, sampling_strategy='minority')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ----------- Normalisation ----------- #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# ----------- Modèle Random Forest ----------- #
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=4, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train_scaled, y_train_resampled)

# ----------- Sélection de variables ----------- #
selector = SelectFromModel(rf_model, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print(f"\nNombre de variables sélectionnées : {len(selected_features)}")
print(f"Variables importantes : {list(selected_features)}")

# ----------- Réentraînement ----------- #
final_rf = RandomForestClassifier(
    n_estimators=200, max_depth=4, random_state=42, class_weight="balanced"
)
final_rf.fit(X_train_selected, y_train_resampled)

# ----------- Probabilités + seuil optimal ----------- #
y_proba = final_rf.predict_proba(X_test_selected)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx]

print(f"\n🔍 Seuil optimal (F1-max): {optimal_threshold:.4f}")

# ----------- Prédiction finale ----------- #
y_pred_final = (y_proba >= optimal_threshold).astype(int)

print("\n### 📊 Résultats finaux Random Forest (seuil optimisé) ###")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print("Matrice de confusion:\n", confusion_matrix(y_test, y_pred_final))
print("Rapport de classification:\n", classification_report(y_test, y_pred_final, digits=4))

# Interprétation de la matrice de confusion
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
print(f"\n✅ TP : {tp}, ❌ FP : {fp}, ❌ FN : {fn}, ✅ TN : {tn}")
print(f"📈 Ratio TP / FP : {tp / fp if fp else 'Infinity'}")

# Matrice normalisée
conf_matrix = confusion_matrix(y_test, y_pred_final)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum()
print("\nMatrice de confusion normalisée :")
print(conf_matrix_normalized)

# Nombre total d’échantillons test
print(f"\nNombre total d’échantillons test : {len(y_test)}")