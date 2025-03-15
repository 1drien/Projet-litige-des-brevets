import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = 'Dataset_Thuy (1).csv'
df = pd.read_csv(file_path)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove "Year_Litigation" to prevent data leakage
if "Year_Litigation" in df.columns:
    df = df.drop(columns=["Year_Litigation"])
    print("\nRemoved 'Year_Litigation' to prevent data leakage.")

# Remove "foreign_priority" to reduce feature bias
if "foreign_priority" in df.columns:
    df = df.drop(columns=["foreign_priority"])
    print("Removed 'foreign_priority' to reduce feature bias.")

# Frequency encoding for categorical variables
for col in df.select_dtypes(include='object').columns:
    freq = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq)

# Split target variable
y = df["Infringment"]
X = df.drop(columns=["Infringment"])

# Split data into train/test BEFORE applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE **only on training data** to balance classes
print("\nClass distribution in TRAIN BEFORE SMOTE:")
print(y_train.value_counts())

smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nClass distribution in TRAIN AFTER SMOTE:")
print(y_train_resampled.value_counts())

print("\nClass distribution in TEST (unchanged):")
print(y_test.value_counts())

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model with improved parameters
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=300,
    learning_rate=0.002,
    max_depth=1,
    min_child_weight=30,
    gamma=10,
    subsample=0.4,
    colsample_bytree=0.5,
    scale_pos_weight=15,
    reg_lambda=10,
    reg_alpha=5,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train_resampled)

# Get probability predictions
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Adjust classification threshold
threshold = 0.6
y_pred = (y_pred_proba >= threshold).astype(int)

# Predictions and evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, digits=4)

print("\n### Results with Improved XGBoost ###")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Extract confusion matrix values
tn, fp, fn, tp = conf_matrix.ravel()

# Print interpreted confusion matrix
print("\nInterpreted Confusion Matrix:")
print(f"True Positives (TP): {tp}  → Correctly predicted litigation cases")
print(f"False Positives (FP): {fp}  → Non-litigation patents incorrectly classified as litigation")
print(f"False Negatives (FN): {fn}  → Litigation patents incorrectly classified as non-litigation")
print(f"True Negatives (TN): {tn}  → Correctly predicted non-litigation patents")

# Compute TP/FP ratio
if fp > 0:
    tp_fp_ratio = tp / fp
else:
    tp_fp_ratio = "Infinity"

print(f"\nRatio TP / FP: {tp_fp_ratio}")

# Normalized confusion matrix for better interpretation
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum()
print("\nNormalized Confusion Matrix (Proportions):")
print(conf_matrix_normalized)

# Feature importance
feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = X.columns[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:10], feature_importance[sorted_idx][:10])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Most Important Features")
plt.gca().invert_yaxis()
plt.show()

print("\nTop 10 Most Important Features:")
for i in range(10):
    print(f"{sorted_features[i]}: {feature_importance[sorted_idx][i]:.4f}")
