import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# Function to optimize memory usage
def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'int64':
            df[col] = df[col].astype(np.int32)
        elif col_type == 'float64':
            df[col] = df[col].astype(np.float32)
        elif col_type == 'object':
            df[col] = df[col].astype('category')
    return df

# Load and optimize the dataset
file_path = 'Dataset_Thuy (1).csv'
df = pd.read_csv(file_path)
df = optimize_memory(df)

# Handle missing values (median imputation)
df.fillna(df.median(numeric_only=True), inplace=True)

# Sample 10% of the dataset to reduce memory usage
df_sample = df.sample(frac=0.1, random_state=42)

# Frequency encoding for categorical variables
for col in df_sample.select_dtypes(include='category').columns:
    freq = df_sample[col].value_counts(normalize=True)
    df_sample[col] = df_sample[col].map(freq)

# Split features (X) and target variable (y)
y = df_sample["Infringment"]
X = df_sample.drop(columns=["Infringment"])

# Display initial class distribution
print("Class distribution before SMOTE:")
print(y.value_counts())

# Split into training and testing sets before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE only to the training set
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Display class distribution after SMOTE (only for training)
print("\nClass distribution after SMOTE (Training Set Only):")
print(pd.Series(y_train_resampled).value_counts())

# Display unchanged test set distribution
print("\nClass distribution in Test Set (Unchanged):")
print(pd.Series(y_test).value_counts())

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model with higher weight for litigation cases
model = LogisticRegression(max_iter=1000, solver='saga', class_weight={0:1, 1:5})
model.fit(X_train_scaled, y_train_resampled)

# Feature selection using a stricter threshold
selector = SelectFromModel(model, threshold="2*mean", prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f"\nNumber of selected features: {len(selected_features)}")
print(f"Important features: {list(selected_features)}")

# Retrain the model using only selected features
final_model = LogisticRegression(max_iter=1000, solver='saga', class_weight={0:1, 1:5})
final_model.fit(X_train_selected, y_train_resampled)

# Adjust classification threshold
threshold = 0.  # Adjust this value to fine-tune predictions
y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
y_pred = (y_pred_proba >= threshold).astype(int)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, digits=4)

# Display results
print("\n### Results after strict feature selection ###")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Extract confusion matrix values
tn, fp, fn, tp = conf_matrix.ravel()

# Print an interpreted confusion matrix
print("\nInterpreted Confusion Matrix:")
print(f"True Positives (TP): {tp}  → Correctly predicted litigation cases")
print(f"False Positives (FP): {fp}  → Non-litigation patents incorrectly classified as litigation")
print(f"False Negatives (FN): {fn}  → Litigation patents incorrectly classified as non-litigation")
print(f"True Negatives (TN): {tn}  → Correctly predicted non-litigation patents")

# Normalize confusion matrix for better interpretation
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum()
print("\nNormalized Confusion Matrix (Proportions):")
print(conf_matrix_normalized)

# Ensure total samples match
print(f"\nTotal samples in y_test: {len(y_test)}")
print(f"Sum of all values in the confusion matrix: {conf_matrix.sum()}")

