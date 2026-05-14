import re
import random
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ Load Data ------------------
def resolve_base_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent]
    for candidate in candidates:
        if (candidate / "Data").exists() and (candidate / "Master Data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find 'Data' and 'Master Data' folders next to test.py or one level above."
    )

BASE_DIR = resolve_base_dir()
training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")
testing  = pd.read_csv(BASE_DIR / "Data" / "Testing.csv")

# ------------------ Dataset Preprocessing ------------------
# Step 1: Clean duplicate column names (e.g. "fever.1" -> "fever")
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns  = testing.columns.str.replace(r"\.\d+$", "", regex=True)

# Step 2: Remove duplicate columns
training = training.loc[:, ~training.columns.duplicated()]
testing  = testing.loc[:,  ~testing.columns.duplicated()]

# Step 3: Fill missing values with 0 (no symptom = 0)
training.fillna(0, inplace=True)
testing.fillna(0, inplace=True)

# Step 4: Features and labels
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Step 5: Encode target labels (disease names -> numbers)
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 6: Train-test split (67% train, 33% test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.33, random_state=42
)

print("\n📦 Dataset Preprocessing Complete:")
print(f"   Total samples    : {len(x)}")
print(f"   Training samples : {len(x_train)}")
print(f"   Testing samples  : {len(x_test)}")
print(f"   Total features   : {len(cols)} symptoms")
print(f"   Total diseases   : {len(le.classes_)}")

# ------------------ KNN Model ------------------
# K-Nearest Neighbors (Supervised Learning):
# When a patient describes symptoms, KNN finds the K=5 most
# similar patients from training data and predicts the disease
# by majority vote among those 5 neighbors.
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(x_train, y_train)
knn_accuracy = round(knn_model.score(x_test[:200], y_test[:200]) * 100, 2)

print(f"\n📊 KNN Model Training Complete:")
print(f"   📍 KNN (K=5) Accuracy : {knn_accuracy}%\n")

# ------------------ Dictionaries ------------------
severityDictionary   = {}
description_list     = {}
precautionDictionary = {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

def getDescription():
    with open(BASE_DIR / "Master Data" / "symptom_Description.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 2:
                description_list[row[0]] = row[1]

def getSeverityDict():
    with open(BASE_DIR / "Master Data" / "Symptom_severity.csv") as csv_file:
        for row in csv.reader(csv_file):
            try:
                severityDictionary[row[0]] = int(row[1])
            except:
                pass

def getprecautionDict():
    with open(BASE_DIR / "Master Data" / "symptom_precaution.csv") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# ------------------ Symptom Synonyms ------------------
symptom_synonyms = {
    "stomach ache"       : "stomach_pain",
    "belly pain"         : "stomach_pain",
    "tummy pain"         : "stomach_pain",
    "loose motion"       : "diarrhea",
    "motions"            : "diarrhea",
    "high temperature"   : "fever",
    "temperature"        : "fever",
    "feaver"             : "fever",
    "coughing"           : "cough",
    "throat pain"        : "sore_throat",
    "cold"               : "chills",
    "breathing issue"    : "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache"          : "muscle_pain",
}

# ------------------ Symptom Extractor ------------------
def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")

    # 1. Synonym replacement
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    # 2. Exact match
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    # 3. Fuzzy match (handles typos)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(
            word,
            [s.replace("_", " ") for s in all_symptoms],
            n=1, cutoff=0.8
        )
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

# ------------------ Build Input Vector ------------------
def build_input_vector(symptoms_list):
    """
    Converts symptom list into a binary feature vector.
    Each position: 1 = symptom present, 0 = absent.
    Returned as DataFrame to match training data format.
    """
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    return pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))

# ------------------ KNN Prediction ------------------
def predict_disease(symptoms_list):
    """
    KNN Prediction:
    Builds a binary symptom vector, then KNN finds the 5
    most similar training samples and votes on the disease.
    Also returns confidence as the winning vote percentage.
    """
    input_vector = build_input_vector(symptoms_list)
    pred_proba   = knn_model.predict_proba(input_vector)[0]
    pred_class   = np.argmax(pred_proba)
    disease      = le.inverse_transform([pred_class])[0]
    confidence   = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence

# ------------------ Severity Calculator ------------------
def calculate_severity(symptoms_list, num_days):
    total_severity = sum(severityDictionary.get(s, 0) for s in symptoms_list)
    score = (total_severity * num_days) / (len(symptoms_list) + 1)
    return round(score, 2)

# ------------------ Empathy Quotes ------------------
quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# ------------------ Chatbot ------------------
def chatbot():
    getSeverityDict()
    getDescription()
    getprecautionDict()

    print("=" * 55)
    print("🤖  Welcome to HealthCare ChatBot")
    print("=" * 55)
    print("Hello! Please answer a few questions so I can")
    print("understand your condition better.\n")

    # -------- Basic Info --------
    name   = input("👉 What is your name?              : ")
    age    = input("👉 Please enter your age           : ")
    gender = input("👉 What is your gender? (M/F/Other): ")

    # -------- Symptom Input --------
    symptoms_input = input(
        "\n👉 Describe your symptoms in a sentence\n"
        "   (e.g. 'I have fever and stomach pain'): "
    )
    symptoms_list = extract_symptoms(symptoms_input, cols)

    if not symptoms_list:
        print("\n❌ Sorry, I could not detect valid symptoms.")
        print("   Please try again with more details.")
        return

    print(f"\n✅ Detected symptoms: {', '.join(symptoms_list)}")

    # -------- Follow-up Questions --------
    num_days       = int(input("👉 For how many days have you had these symptoms?       : "))
    severity_scale = int(input("👉 On a scale of 1-10, how severe is your condition?   : "))
    pre_exist      = input("👉 Any pre-existing conditions (diabetes, hypertension)? : ")
    lifestyle      = input("👉 Do you smoke, drink alcohol, or have irregular sleep? : ")
    family         = input("👉 Any family history of similar illness?                : ")

    # -------- Initial Prediction for guided questions --------
    disease_initial, _ = predict_disease(symptoms_list)

    # -------- Guided Disease-Specific Questions --------
    print(f"\n🤔 Let me ask a few more questions about {disease_initial}...")
    disease_row = training[training['prognosis'] == disease_initial]
    if not disease_row.empty:
        disease_symptoms = list(
            disease_row.iloc[0][:-1].index[
                disease_row.iloc[0][:-1] == 1
            ]
        )
        asked = 0
        for sym in disease_symptoms:
            if sym not in symptoms_list and asked < 8:
                ans = input(
                    f"👉 Do you also have {sym.replace('_', ' ')}? (yes/no): "
                ).strip().lower()
                if ans == "yes":
                    symptoms_list.append(sym)
                asked += 1

    # -------- Final KNN Prediction --------
    disease, confidence = predict_disease(symptoms_list)
    severity_score      = calculate_severity(symptoms_list, num_days)

    # -------- Print Results --------
    print("\n" + "=" * 55)
    print("🩺  DIAGNOSIS REPORT")
    print("=" * 55)

    print(f"\n👤 Patient  : {name} | Age: {age} | Gender: {gender}")
    print(f"📅 Duration : {num_days} days")
    print(f"⚡ Severity : {severity_scale}/10 (self-reported) | Score: {severity_score}")
    print(f"🧬 Symptoms : {', '.join(symptoms_list)}")

    print(f"\n--- KNN Prediction (K=5) ---")
    print(f"📍 Predicted Disease : {disease}")
    print(f"🔎 Confidence        : {confidence}%")

    print(f"\n📖 About {disease}:")
    print(f"   {description_list.get(disease, 'No description available.')}")

    if disease in precautionDictionary:
        print(f"\n🛡️  Suggested Precautions:")
        for i, prec in enumerate(precautionDictionary[disease], 1):
            print(f"   {i}. {prec}")

    # Severity warning
    if severity_score > 13 or severity_scale >= 8:
        print("\n🚨 WARNING: Your symptoms appear severe.")
        print("   Please consult a doctor immediately.")
    elif severity_score > 5:
        print("\n⚠️  Your condition seems moderate. Please visit a clinic soon.")
    else:
        print("\n✅ Your condition seems mild. Rest and monitor your symptoms.")

    print("\n💡 " + random.choice(quotes))
    print(f"\nThank you for using HealthCare ChatBot.")
    print(f"Wishing you good health, {name}! 💚")
    print("=" * 55)

# ------------------ Run ------------------
if __name__ == "__main__":
    chatbot()