import joblib
import numpy as np

# Load model
model = joblib.load("model_diagnosa.pkl")

print("=" * 50)
print("🎯 Model Information")
print("=" * 50)
print("Classes (Diseases):", model.classes_)
print("Number of features:", model.n_features_in_)
print()

# Test prediction
print("=" * 50)
print("🧪 Test Predictions")
print("=" * 50)

test_cases = [
    {"name": "ISPA", "data": [1, 1, 0, 0, 0, 0, 1, 0]},
    {"name": "DBD", "data": [1, 0, 1, 0, 0, 1, 0, 0]},
    {"name": "GASTRITIS", "data": [0, 0, 0, 1, 1, 0, 0, 0]},
    {"name": "HIPERTENSI", "data": [0, 0, 0, 0, 0, 1, 0, 1]},
    {"name": "PNEUMONIA", "data": [1, 1, 0, 0, 0, 0, 1, 0]},
    {"name": "MALARIA", "data": [1, 0, 1, 1, 0, 1, 0, 0]},
    {"name": "DIARE", "data": [0, 0, 0, 1, 1, 0, 0, 0]},
    {"name": "ASMA", "data": [0, 1, 0, 0, 0, 0, 1, 0]},
    {"name": "DIABETES", "data": [0, 0, 0, 0, 0, 0, 0, 1]},
    {"name": "TIFUS", "data": [1, 0, 0, 1, 1, 0, 0, 0]},
]

for test in test_cases:
    test_data = np.array([test["data"]])
    prediction = model.predict(test_data)
    probability = model.predict_proba(test_data).max()
    print(f"Input: {test['data']}")
    print(f"  → Prediction: {prediction[0]} (confidence: {probability*100:.1f}%)")
    print()
