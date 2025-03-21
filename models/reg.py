import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# ----------- Chargement et Prétraitement ----------- #
file_path = 'Dataset_Thuy (1).csv'
df = pd.read_csv(file_path)

# Conversion optimisée des types
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype(np.int32)
    elif df[col].dtype == 'float64':
        df[col] = df[col].astype(np.float32)
    elif df[col].dtype == 'object':
        df[col] = df[col].astype('category')

# Imputation des valeurs manquantes
df.fillna(df.median(numeric_only=True), inplace=True)

# Fréquence encoding des variables catégorielles
for col in df.select_dtypes(include='category').columns:
    freq = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq)

# Ciblage et séparation des variables
y = df["Infringment"]
X = df.drop(columns=["Infringment"])

# Split initial
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ----------- Rééquilibrage avec SMOTE ----------- #
print("\nClass distribution before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# ----------- Standardisation ----------- #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# ----------- Régression logistique avec pondération ----------- #
model = LogisticRegression(max_iter=1000, solver='saga', class_weight={0: 1, 1: 5})
model.fit(X_train_scaled, y_train_resampled)

# ----------- Sélection de variables modérée ----------- #
selector = SelectFromModel(model, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
selected_features = X.columns[selector.get_support()]
print(f"\nNombre de variables sélectionnées : {len(selected_features)}")
print(f"Variables importantes : {list(selected_features)}")

# ----------- Modèle final ----------- #
final_model = LogisticRegression(max_iter=1000, solver='saga', class_weight={0: 1, 1: 5})
final_model.fit(X_train_selected, y_train_resampled)

# ----------- Seuil optimal via courbe précision/rappel ----------- #
y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx]
print(f"\nSeuil optimal (F1-max) : {optimal_threshold:.4f}")

# ----------- Prédictions et évaluation ----------- #
y_pred = (y_pred_proba >= optimal_threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, digits=4)

print("\n### Résultats du modèle de régression logistique optimisé ###")
print(f"Accuracy : {accuracy:.4f}")
print("Matrice de confusion :\n", conf_matrix)
print("Rapport de classification :\n", class_report)

# Interprétation confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()
print(f"\nTP (Litiges bien détectés)   : {tp}")
print(f"FP (Faux positifs)            : {fp}")
print(f"FN (Litiges manqués)          : {fn}")
print(f"TN (Non-litiges bien exclus)  : {tn}")

# Ratio TP / FP
tp_fp_ratio = tp / fp if fp > 0 else "Infinity"
print(f"\nRatio TP / FP : {tp_fp_ratio}")

# Matrice normalisée
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum()
print("\nMatrice de confusion normalisée :")
print(conf_matrix_normalized)

# Nombre total d’échantillons test
print(f"\nNombre total d’échantillons dans y_test : {len(y_test)}")
