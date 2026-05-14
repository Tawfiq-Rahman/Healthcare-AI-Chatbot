import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


def train_knn(x_train, y_train, x_test, y_test):
    """
    Trains a KNN classifier with K=5 neighbors.

    How KNN works:
    When a new patient comes in, KNN looks at the 5 most
    similar patients in training data (based on symptom pattern)
    and predicts the disease by majority vote among those 5.

    Returns:
        knn_model, knn_accuracy
    """
    knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn_model.fit(x_train, y_train)

    # Evaluate on a subset to avoid slowness
    knn_accuracy = round(knn_model.score(x_test[:200], y_test[:200]) * 100, 2)

    print(f"\n📊 KNN Model Training Complete:")
    print(f"   📍 KNN (K=5) Accuracy : {knn_accuracy}%\n")

    return knn_model, knn_accuracy


def build_input_vector(symptoms_list, symptoms_dict, cols):
    """
    Converts a list of symptom strings into a binary feature vector.
    Each position represents one symptom: 1 = present, 0 = absent.
    Returned as DataFrame to match training data format (avoids sklearn warning).
    """
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    return pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))


def predict_disease(symptoms_list, knn_model, le, symptoms_dict, cols, training):
    """
    KNN Prediction:
    1. Builds binary symptom vector from patient input
    2. KNN finds 5 most similar training records
    3. Returns predicted disease + realistic confidence score

    Confidence is calculated using 3 factors:
    - Neighbor agreement: how many of 5 neighbors agree (KNN proba)
    - Symptom coverage: how many of disease's known symptoms were reported
    - Count penalty: fewer symptoms = less certainty
    """
    input_vector = build_input_vector(symptoms_list, symptoms_dict, cols)

    pred_proba  = knn_model.predict_proba(input_vector)[0]
    pred_class  = np.argmax(pred_proba)
    disease     = le.inverse_transform([pred_class])[0]
    raw_conf    = pred_proba[pred_class]

    # Factor 1: Symptom coverage ratio
    disease_row       = training[training['prognosis'] == disease]
    total_disease_sym = int(disease_row.iloc[0][:-1].sum()) if not disease_row.empty else 1
    reported_count    = len(symptoms_list)
    coverage_ratio    = min(reported_count / max(total_disease_sym, 1), 1.0)

    # Factor 2: Neighbor agreement (raw KNN probability)
    neighbor_agreement = raw_conf

    # Factor 3: Count penalty — fewer symptoms = lower confidence
    if reported_count <= 2:
        count_factor = 0.55
    elif reported_count <= 4:
        count_factor = 0.75
    else:
        count_factor = 1.0

    # Weighted blend
    blended    = (0.5 * neighbor_agreement) + (0.5 * coverage_ratio)
    confidence = round(blended * count_factor * 100, 1)

    # Clamp to realistic range [35, 95]
    confidence = max(35.0, min(confidence, 95.0))

    return disease, confidence