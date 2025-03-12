import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# Memory Optimization Function
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

# Verification of the target variable
print("Target variable used for Logistic Regression:", y.name)
print("Class distribution before SMOTE:")
print(y.value_counts())

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verify class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Split data into train/test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model with adjusted class weights
model = LogisticRegression(max_iter=1000, solver='saga', class_weight={0:1, 1:2.5})
model.fit(X_train_scaled, y_train)

# Feature selection using SelectFromModel
selector = SelectFromModel(model, threshold="1.5*mean", prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Display selected features
selected_features = X.columns[selector.get_support()]
print(f"\nNumber of selected features (strict threshold): {len(selected_features)}")
print(f"Important features: {list(selected_features)}")

# Retrain the model with only selected features
final_model = LogisticRegression(max_iter=1000, solver='saga', class_weight={0:1, 1:2.5})
final_model.fit(X_train_selected, y_train)

# Predictions and Evaluation
y_pred = final_model.predict(X_test_selected)

# Results after strict feature selection
print("\n### Results after strict feature selection ###")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
