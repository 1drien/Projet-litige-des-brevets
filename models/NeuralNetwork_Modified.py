
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ------------------- Chargement du dataset ------------------- #
df = pd.read_csv("Dataset.csv")

# ------------------- Prétraitement ------------------- #
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype(np.int32)
    elif df[col].dtype == 'float64':
        df[col] = df[col].astype(np.float32)
    elif df[col].dtype == 'object':
        df[col] = df[col].astype('category')

df.fillna(df.median(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='category').columns:
    freq = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq)

y = df["Infringment"]
X = df.drop(columns=["Infringment"]).values

# ------------------- Cross-validation ------------------- #
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in skf.split(X, y):
    print(f"\n--- Fold {fold} ---")

    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold = y.iloc[train_index].values
    y_test_fold = y.iloc[test_index].values

    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_train_fold_res, y_train_fold_res = smote.fit_resample(X_train_fold, y_train_fold)

    scaler = StandardScaler()
    X_train_fold_scaled = scaler.fit_transform(X_train_fold_res)
    X_test_fold_scaled = scaler.transform(X_test_fold)

    def build_model(input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = build_model(X_train_fold_scaled.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train_fold_scaled, y_train_fold_res,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred_proba = model.predict(X_test_fold_scaled).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    conf_matrix = confusion_matrix(y_test_fold, y_pred)
    accuracy = accuracy_score(y_test_fold, y_pred)
    class_report = classification_report(y_test_fold, y_pred, digits=4)

    print(f"\n### Résultats du modèle Réseau de Neurones 64 32 16 1 — Fold {fold} ###")
    print(f"Accuracy : {accuracy:.4f}")
    print("Matrice de confusion :\n", conf_matrix)
    print("Rapport de classification :\n", class_report)

    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"\nTP (Litiges bien détectés)   : {tp}")
    print(f"FP (Faux positifs)            : {fp}")
    print(f"FN (Litiges manqués)          : {fn}")
    print(f"TN (Non-litiges bien exclus)  : {tn}")

    tp_fp_ratio = tp / fp if fp > 0 else "Infinity"
    print(f"\nRatio TP / FP : {tp_fp_ratio}")

    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum()
    print("\nMatrice de confusion normalisée :")
    print(conf_matrix_normalized)

    print(f"\nNombre total d’échantillons dans y_test : {len(y_test_fold)}")

    model.save(f"model_fold_{fold}.h5")
    print(f"Modèle fold {fold} sauvegardé sous 'model_fold_{fold}.h5'.")

    fold += 1
