import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. LOADING THE DATASET
# ==========================================
# Replace 'heart_data.csv' with the actual path to your file
try:
    df = pd.read_csv('heart_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    # Creating dummy data matching your columns just so the script can run seamlessly
    print("Local file not found. Creating a placeholder dataframe for demonstration...")
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    dummy_data = np.random.randint(0, 2, size=(100, len(columns)))
    df = pd.DataFrame(dummy_data, columns=columns)

# Preview the first 5 rows
print("\n--- Dataset Preview ---")
print(df.head())

# ==========================================
# 2. SPLITTING DATA INTO FEATURES & TARGET
# ==========================================
# 'X' contains all independent variables (features)
X = df.drop(columns=['target'])

# 'y' contains the dependent variable we want to predict
y = df['target']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ==========================================
# 3. FEATURE SCALING
# ==========================================
# Features like 'chol' (cholesterol) and 'trestbps' (blood pressure) have much higher 
# ranges than binary features like 'sex' or 'fbs'. We scale them so the model treats them fairly.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. TRAINING THE MODEL
# ==========================================
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("\nModel training complete.")

# ==========================================
# 5. EVALUATING THE MODEL
# ==========================================
y_pred = model.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 6. MAKING A PREDICTION ON NEW DATA
# ==========================================
print("\n--- Making a Prediction on a New Patient ---")

# Let's create a hypothetical patient profile matching your exact columns:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
new_patient = np.array([[57, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

# Crucial Step: Scale the new patient data using the *same* scaler we used for training
new_patient_scaled = scaler.transform(new_patient)

# Predict class (0 or 1)
prediction = model.predict(new_patient_scaled)

# Predict probability
probability = model.predict_proba(new_patient_scaled)

if prediction[0] == 1:
    print(f"Result: High risk of heart disease (Probability: {probability[0][1]:.2%})")
else:
    print(f"Result: Low risk of heart disease (Probability: {probability[0][0]:.2%})")
