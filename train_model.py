import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Daftar penyakit yang dapat didiagnosis (25 penyakit)
diseases = [
    "ISPA", "DBD", "TIFUS", "GASTRITIS", "HIPERTENSI",
    "PNEUMONIA", "MALARIA", "DIARE", "ASMA", "DIABETES",
    "TBC", "HEPATITIS", "GINJAL", "JANTUNG", "ANEMIA",
    "ASAM URAT", "VERTIGO", "BRONKITIS", "DEMAM BERDARAH", "LEPTOSPIROSIS",
    "COVID19", "INFLUENZA", "MIOKARDITIS", "APPENDISITIS", "PANKREATITIS"
]

def generate_data():
    """Generate data latih yang lebih banyak dan bervariasi"""
    data = []

    disease_patterns = {
        "ISPA": {"Demam": 0.8, "Batuk": 0.9, "NyeriOtot": 0.3, "Mual": 0.2, "NyeriPerut": 0.1, "Pusing": 0.5, "Sesak": 0.4, "TensiTinggi": 0.1},
        "DBD": {"Demam": 0.95, "Batuk": 0.2, "NyeriOtot": 0.8, "Mual": 0.5, "NyeriPerut": 0.3, "Pusing": 0.9, "Sesak": 0.1, "TensiTinggi": 0.1},
        "TIFUS": {"Demam": 0.9, "Batuk": 0.2, "NyeriOtot": 0.4, "Mual": 0.8, "NyeriPerut": 0.7, "Pusing": 0.6, "Sesak": 0.1, "TensiTinggi": 0.1},
        "GASTRITIS": {"Demam": 0.2, "Batuk": 0.1, "NyeriOtot": 0.2, "Mual": 0.9, "NyeriPerut": 0.9, "Pusing": 0.3, "Sesak": 0.1, "TensiTinggi": 0.1},
        "HIPERTENSI": {"Demam": 0.1, "Batuk": 0.1, "NyeriOtot": 0.2, "Mual": 0.2, "NyeriPerut": 0.1, "Pusing": 0.8, "Sesak": 0.3, "TensiTinggi": 0.95},
        "PNEUMONIA": {"Demam": 0.9, "Batuk": 0.95, "NyeriOtot": 0.4, "Mual": 0.2, "NyeriPerut": 0.1, "Pusing": 0.3, "Sesak": 0.9, "TensiTinggi": 0.2},
        "MALARIA": {"Demam": 0.95, "Batuk": 0.1, "NyeriOtot": 0.7, "Mual": 0.6, "NyeriPerut": 0.4, "Pusing": 0.8, "Sesak": 0.2, "TensiTinggi": 0.1},
        "DIARE": {"Demam": 0.4, "Batuk": 0.1, "NyeriOtot": 0.2, "Mual": 0.8, "NyeriPerut": 0.9, "Pusing": 0.3, "Sesak": 0.1, "TensiTinggi": 0.1},
        "ASMA": {"Demam": 0.2, "Batuk": 0.7, "NyeriOtot": 0.1, "Mual": 0.1, "NyeriPerut": 0.1, "Pusing": 0.2, "Sesak": 0.95, "TensiTinggi": 0.1},
        "DIABETES": {"Demam": 0.2, "Batuk": 0.1, "NyeriOtot": 0.3, "Mual": 0.4, "NyeriPerut": 0.2, "Pusing": 0.3, "Sesak": 0.2, "TensiTinggi": 0.6},
        "TBC": {"Demam": 0.7, "Batuk": 0.95, "NyeriOtot": 0.3, "Mual": 0.2, "NyeriPerut": 0.1, "Pusing": 0.4, "Sesak": 0.8, "TensiTinggi": 0.1},
        "HEPATITIS": {"Demam": 0.6, "Batuk": 0.1, "NyeriOtot": 0.4, "Mual": 0.8, "NyeriPerut": 0.7, "Pusing": 0.3, "Sesak": 0.1, "TensiTinggi": 0.1},
        "GINJAL": {"Demam": 0.5, "Batuk": 0.1, "NyeriOtot": 0.3, "Mual": 0.6, "NyeriPerut": 0.4, "Pusing": 0.5, "Sesak": 0.3, "TensiTinggi": 0.7},
        "JANTUNG": {"Demam": 0.2, "Batuk": 0.3, "NyeriOtot": 0.3, "Mual": 0.3, "NyeriPerut": 0.4, "Pusing": 0.5, "Sesak": 0.8, "TensiTinggi": 0.8},
        "ANEMIA": {"Demam": 0.3, "Batuk": 0.1, "NyeriOtot": 0.4, "Mual": 0.2, "NyeriPerut": 0.1, "Pusing": 0.8, "Sesak": 0.4, "TensiTinggi": 0.2},
        "ASAM URAT": {"Demam": 0.3, "Batuk": 0.1, "NyeriOtot": 0.9, "Mual": 0.2, "NyeriPerut": 0.3, "Pusing": 0.4, "Sesak": 0.1, "TensiTinggi": 0.3},
        "VERTIGO": {"Demam": 0.1, "Batuk": 0.1, "NyeriOtot": 0.2, "Mual": 0.6, "NyeriPerut": 0.1, "Pusing": 0.95, "Sesak": 0.2, "TensiTinggi": 0.3},
        "BRONKITIS": {"Demam": 0.5, "Batuk": 0.95, "NyeriOtot": 0.2, "Mual": 0.1, "NyeriPerut": 0.1, "Pusing": 0.3, "Sesak": 0.6, "TensiTinggi": 0.1},
        "DEMAM BERDARAH": {"Demam": 0.95, "Batuk": 0.2, "NyeriOtot": 0.7, "Mual": 0.5, "NyeriPerut": 0.4, "Pusing": 0.8, "Sesak": 0.1, "TensiTinggi": 0.2},
        "LEPTOSPIROSIS": {"Demam": 0.9, "Batuk": 0.2, "NyeriOtot": 0.8, "Mual": 0.6, "NyeriPerut": 0.5, "Pusing": 0.7, "Sesak": 0.2, "TensiTinggi": 0.2},
        "COVID19": {"Demam": 0.8, "Batuk": 0.7, "NyeriOtot": 0.5, "Mual": 0.3, "NyeriPerut": 0.2, "Pusing": 0.5, "Sesak": 0.7, "TensiTinggi": 0.2},
        "INFLUENZA": {"Demam": 0.85, "Batuk": 0.8, "NyeriOtot": 0.6, "Mual": 0.3, "NyeriPerut": 0.1, "Pusing": 0.6, "Sesak": 0.3, "TensiTinggi": 0.1},
        "MIOKARDITIS": {"Demam": 0.5, "Batuk": 0.2, "NyeriOtot": 0.5, "Mual": 0.3, "NyeriPerut": 0.3, "Pusing": 0.4, "Sesak": 0.7, "TensiTinggi": 0.6},
        "APPENDISITIS": {"Demam": 0.6, "Batuk": 0.1, "NyeriOtot": 0.3, "Mual": 0.5, "NyeriPerut": 0.95, "Pusing": 0.3, "Sesak": 0.1, "TensiTinggi": 0.2},
        "PANKREATITIS": {"Demam": 0.5, "Batuk": 0.1, "NyeriOtot": 0.4, "Mual": 0.8, "NyeriPerut": 0.9, "Pusing": 0.4, "Sesak": 0.2, "TensiTinggi": 0.2}
    }

    # Generate lebih banyak data (50,000 samples - 25 diseases x 2000)
    for disease in diseases:
        pattern = disease_patterns[disease]
        samples_per_disease = 2000  # 25 diseases x 2000 = 50,000 samples
        
        for _ in range(samples_per_disease):
            symptoms = {}
            for symptom, prob in pattern.items():
                # Tambah noise untuk variasi
                adjusted_prob = prob + random.uniform(-0.2, 0.2)
                adjusted_prob = max(0, min(1, adjusted_prob))
                symptoms[symptom] = 1 if random.random() < adjusted_prob else 0
            
            # Ensure at least 1 symptom
            if sum(symptoms.values()) == 0:
                key = random.choice(list(pattern.keys()))
                symptoms[key] = 1
            
            data.append([
                symptoms["Demam"], symptoms["Batuk"], symptoms["NyeriOtot"], symptoms["Mual"],
                symptoms["NyeriPerut"], symptoms["Pusing"], symptoms["Sesak"],
                symptoms["TensiTinggi"], disease
            ])

    random.shuffle(data)

    columns = ["Demam", "Batuk", "NyeriOtot", "Mual", "NyeriPerut", "Pusing", "Sesak", "TensiTinggi", "Diagnosis"]
    return pd.DataFrame(data, columns=columns)


# Generate dataset
print("=" * 60)
print("Generating 50,000 training samples (25 diseases)...")
print("=" * 60)
df = generate_data()
print(f"Total samples: {len(df)}")
print(f"\nDisease distribution:\n{df['Diagnosis'].value_counts()}")

# Split data
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" + "=" * 60)
print("Training Multiple Models...")
print("=" * 60)

# Model 1: RandomForest
print("\n[1/3] Training RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   RandomForest Accuracy: {rf_acc:.2%}")

# Model 2: GradientBoosting
print("\n[2/3] Training GradientBoosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"   GradientBoosting Accuracy: {gb_acc:.2%}")

# Model 3: Neural Network (MLP)
print("\n[3/3] Training Neural Network...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_model.fit(X_train, y_train)
mlp_pred = mlp_model.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)
print(f"   Neural Network Accuracy: {mlp_acc:.2%}")

# Ensemble: Voting Classifier
print("\n[4/4] Creating Ensemble Model...")
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('mlp', mlp_model)
    ],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"   Ensemble Accuracy: {ensemble_acc:.2%}")

# Choose best model
best_model = ensemble_model
best_acc = ensemble_acc
best_name = "Ensemble (RF + GB + MLP)"

if rf_acc > best_acc:
    best_model = rf_model
    best_acc = rf_acc
    best_name = "RandomForest"
if gb_acc > best_acc:
    best_model = gb_model
    best_acc = gb_acc
    best_name = "GradientBoosting"
if mlp_acc > best_acc:
    best_model = mlp_model
    best_acc = mlp_acc
    best_name = "Neural Network"

print("\n" + "=" * 60)
print(f"Best Model: {best_name} (Accuracy: {best_acc:.2%})")
print("=" * 60)

# Save best model
joblib.dump(best_model, "model_diagnosa.pkl")
print(f"\nModel saved to 'model_diagnosa.pkl'")

# Save all models for comparison
joblib.dump({
    "rf": rf_model,
    "gb": gb_model,
    "mlp": mlp_model,
    "ensemble": ensemble_model,
    "best_name": best_name
}, "all_models.pkl")
print("All models saved to 'all_models.pkl'")

print("\nClassification Report (Best Model):")
print(classification_report(y_test, best_model.predict(X_test)))

print("\nDiseases that can be diagnosed:")
for disease in diseases:
    print(f"   - {disease}")
