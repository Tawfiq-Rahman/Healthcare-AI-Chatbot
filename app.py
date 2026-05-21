import random
from flask import Flask, render_template, request, jsonify, session
import warnings
warnings.filterwarnings("ignore")

# ── Import from our own modules ──────────────────────────────
from preprocessing     import load_and_preprocess
from model             import train_knn, predict_disease
from utils             import load_description, load_severity, load_precautions, calculate_severity
from symptom_extractor import extract_symptoms

# ── App setup ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "healthcare_chatbot_secret"

# ── Load data & train model on startup ───────────────────────
x_train, x_test, y_train, y_test, le, cols, training, BASE_DIR = load_and_preprocess()
knn_model, knn_accuracy = train_knn(x_train, y_train, x_test, y_test)

# Build symptoms index dict
symptoms_dict = {symptom: idx for idx, symptom in enumerate(training[cols.tolist()])}

# ── Load master data dictionaries ────────────────────────────
description_list     = load_description(BASE_DIR)
severity_dict        = load_severity(BASE_DIR)
precaution_dict      = load_precautions(BASE_DIR)

# ── Empathy quotes ───────────────────────────────────────────
quotes = [
    "Health is wealth, take care of yourself.",
    "A healthy outside starts from the inside.",
    "Every day is a chance to get stronger and healthier.",
    "Take a deep breath, your health matters the most.",
    "Remember, self-care is not selfish."
]

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data  = request.json
    step  = data.get("step")
    value = data.get("value", "").strip()

    if step == "name":
        session["name"] = value
        return jsonify({"reply": f"Nice to meet you, {value}! What is your age?", "next_step": "age"})

    elif step == "age":
        session["age"] = value
        return jsonify({"reply": "What is your gender? (M/F/Other)", "next_step": "gender"})

    elif step == "gender":
        session["gender"] = value
        return jsonify({
            "reply": "Please describe your symptoms in a sentence. (e.g. 'I have fever and stomach pain')",
            "next_step": "symptoms"
        })

    elif step == "symptoms":
        symptoms_list = extract_symptoms(value, cols)
        if not symptoms_list:
            return jsonify({
                "reply": "Sorry, I could not detect valid symptoms. Please try again with more details.",
                "next_step": "symptoms"
            })
        session["symptoms"] = symptoms_list
        detected = ", ".join(s.replace("_", " ") for s in symptoms_list)
        return jsonify({
            "reply"   : f"Detected symptoms: {detected}. For how many days have you had these symptoms?",
            "next_step": "days",
            "tag"     : "symptoms_detected",
            "symptoms": detected
        })

    elif step == "days":
        try:
            session["num_days"] = int(value)
        except:
            return jsonify({"reply": "Please enter a valid number of days.", "next_step": "days"})
        return jsonify({"reply": "On a scale of 1–10, how severe is your condition?", "next_step": "severity"})

    elif step == "severity":
        try:
            session["severity_scale"] = int(value)
        except:
            return jsonify({"reply": "Please enter a number between 1 and 10.", "next_step": "severity"})
        return jsonify({
            "reply": "Do you have any pre-existing conditions? (e.g. diabetes, hypertension, or type 'None')",
            "next_step": "pre_exist"
        })

    elif step == "pre_exist":
        session["pre_exist"] = value
        return jsonify({
            "reply": "Do you smoke, drink alcohol, or have irregular sleep? (or type 'None')",
            "next_step": "lifestyle"
        })

    elif step == "lifestyle":
        session["lifestyle"] = value
        return jsonify({
            "reply": "Any family history of similar illness? (or type 'None')",
            "next_step": "family"
        })

    elif step == "family":
        session["family"] = value
        symptoms_list      = session.get("symptoms", [])

        disease_initial, _ = predict_disease(
            symptoms_list, knn_model, le, symptoms_dict, cols, training
        )
        session["disease_initial"] = disease_initial

        disease_row = training[training['prognosis'] == disease_initial]
        guided = []
        if not disease_row.empty:
            ds     = list(disease_row.iloc[0][:-1].index[disease_row.iloc[0][:-1] == 1])
            guided = [s for s in ds if s not in symptoms_list][:8]
        session["guided_queue"] = guided
        session["guided_index"] = 0

        if guided:
            first = guided[0].replace("_", " ")
            return jsonify({
                "reply"    : f"Let me ask a few more questions about {disease_initial}. Do you also have {first}?",
                "next_step": "guided",
                "is_yesno" : True
            })
        return finalize()

    elif step == "guided":
        queue         = session.get("guided_queue", [])
        idx           = session.get("guided_index", 0)
        symptoms_list = session.get("symptoms", [])

        if idx < len(queue):
            if value.lower() == "yes":
                symptoms_list.append(queue[idx])
                session["symptoms"] = symptoms_list
            idx += 1
            session["guided_index"] = idx

        if idx < len(queue):
            next_sym = queue[idx].replace("_", " ")
            return jsonify({
                "reply"    : f"Do you also have {next_sym}?",
                "next_step": "guided",
                "is_yesno" : True
            })
        return finalize()

    return jsonify({"reply": "Something went wrong. Please refresh.", "next_step": "name"})


def finalize():
    """Runs final KNN prediction and returns full diagnosis result."""
    symptoms_list  = session.get("symptoms", [])
    num_days       = session.get("num_days", 1)
    severity_scale = session.get("severity_scale", 1)
    name           = session.get("name", "Friend")

    disease, confidence = predict_disease(
        symptoms_list, knn_model, le, symptoms_dict, cols, training
    )
    severity_score = calculate_severity(symptoms_list, num_days, severity_dict)

    if severity_score > 13 or severity_scale >= 8:
        severity_msg   = "Your symptoms appear SEVERE. Please consult a doctor immediately."
        severity_level = "severe"
    elif severity_score > 5:
        severity_msg   = "Your condition seems moderate. Please visit a clinic soon."
        severity_level = "moderate"
    else:
        severity_msg   = "Your condition seems mild. Rest and monitor your symptoms."
        severity_level = "mild"

    return jsonify({
        "reply"         : "Here is your diagnosis report.",
        "next_step"     : "done",
        "result"        : True,
        "disease"       : disease,
        "confidence"    : confidence,
        "description"   : description_list.get(disease, "No description available."),
        "precautions"   : precaution_dict.get(disease, []),
        "severity_score": severity_score,
        "severity_scale": severity_scale,
        "severity_msg"  : severity_msg,
        "severity_level": severity_level,
        "symptoms"      : [s.replace("_", " ") for s in symptoms_list],
        "num_days"      : num_days,
        "name"          : name,
        "age"           : session.get("age", ""),
        "gender"        : session.get("gender", ""),
        "quote"         : random.choice(quotes),
        "knn_accuracy"  : knn_accuracy
    })


if __name__ == "__main__":
    app.run(debug=True)