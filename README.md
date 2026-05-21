# <h2 align="left"><b>HealthCare AI ChatBot</b></h2>

An interactive healthcare assistant chatbot built using **Flask** and a **K-Nearest Neighbors (KNN)** machine learning model. The bot helps users identify potential health conditions by conversational symptom gathering, predicting the disease with calculated confidence levels, and offering tailored medical descriptions, precautions, and severity assessments.

---

## ✨ Features
* 🤖 **Interactive AI Consultation:** Gathers basic profile data (name, age, gender) and guides users through clarifying questions.
* 🧠 **Symptom Extraction & ML Prediction:** Utilizes a KNN classification engine behind a Python-based preprocessing pipeline to match symptoms to potential prognoses.
* 📊 **Risk & Severity Assessment:** Calculates health risk indicators dynamically based on explicit timelines, severe scales, and historical indicators.
* 🎨 **Modern Chat UI:** Designed with responsive, beautiful CSS components featuring dynamic typing indicators, message delays, interactive Yes/No flows, and animated result summary charts.

---

## 📁 Repository Structure
```text
├── 📁 Data/                   # Datasets for model training
├── 📁 Master Data/            # Sourced dictionaries for descriptions/precautions
├── 📁 templates/
│   └── 📄 index.html          # Custom Frontend UI/UX Design
├── 📄 app.py                  # Core Flask routing app orchestration
├── 📄 model.py                # KNN Model setup and processing functions
├── 📄 preprocessing.py        # Feature vector & pipeline extraction engine
├── 📄 symptom_extractor.py    # Text parsing logic for user inputs
├── 📄 utils.py                # Severity metrics and file loaders
└── 📄 .gitignore              # Clean version tracking logic
