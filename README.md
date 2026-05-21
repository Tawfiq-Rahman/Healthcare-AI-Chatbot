==============================================
                    AI MEDICAL DIAGNOSIS CHATBOT
                         PROJECT DOCUMENTATION
================================================================================



1. PROJECT OVERVIEW
================================================================================

AI Medical Diagnosis Chatbot is an intelligent healthcare assistance system that 
helps patients self-assess potential diseases based on their symptoms. Using 
machine learning (K-Nearest Neighbors) and natural language processing, the 
chatbot conducts a guided medical interview and provides disease predictions 
with personalized precautions.

KEY FEATURES:
- Multi-turn conversational interface
- NLP-based symptom extraction (handles synonyms, typos, fuzzy matching)
- KNN classifier for disease prediction (K=5 neighbors)
- Confidence scoring based on symptom coverage
- Disease descriptions and preventive precautions
- Severity assessment based on symptom weights

TECHNOLOGY STACK:
- Backend: Flask (Python web framework)
- ML Model: scikit-learn (K-Nearest Neighbors)
- NLP: difflib, regex pattern matching
- Data Processing: pandas, numpy
- Frontend: HTML/JavaScript (vanilla)



2. TEAM MEMBERS & RESPONSIBILITIES
================================================================================

MEMBER A - Project Lead & Documentation
Role: Lead Documentation, Error Handling, Quality Assurance
Days 1-4: README, app.py validation, final polish

MEMBER B - Backend Developer
Role: App Architecture, Model Optimization, Code Cleanup
Days 2-4: Backend documentation, code optimization, performance analysis

MEMBER C - Data & Testing Engineer
Role: Data Pipeline, Testing Framework, Deployment Setup
Days 2-4: Data documentation, test framework, deployment guides



3. PROJECT STRUCTURE
================================================================================

Ai_chatbot/
├── app.py                          # Flask application (Member A & B)
├── model.py                        # KNN classifier (Member B)
├── preprocessing.py                # Data pipeline (Member B)
├── symptom_extractor.py            # NLP symptom extraction (Member C)
├── utils.py                        # Utility functions (Member C)
├── test.py                         # Testing script (Member C)
├── index.html                      # Frontend UI
├── requirements.txt                # Dependencies (Member C)
├── README.md                       # This file
├── Data/
│   ├── dataset.csv                 # Complete dataset
│   ├── Training.csv                # Training data (67% split)
│   └── Testing.csv                 # Testing data (33% split)
└── Master Data/
    ├── symptom_Description.csv     # Disease descriptions
    ├── symptom_precaution.csv      # Disease precautions
    └── Symptom_severity.csv        # Symptom severity weights



4. DATA FLOW ARCHITECTURE
================================================================================

User Input (Web UI)
    |
    v
[app.py] /chat endpoint receives user message
    |
    v
[symptom_extractor.py] Extracts symptoms from free text
    |-- Synonym mapping (e.g., "stomach ache" → "stomach_pain")
    |-- Exact matching of known symptoms
    |-- Fuzzy matching for typos (80% threshold)
    |
    v
[model.py] KNN classifier predicts disease
    |-- Converts symptoms to binary feature vectors
    |-- Finds 5 nearest neighbors
    |-- Calculates confidence score
    |
    v
[utils.py] Loads disease information
    |-- Disease descriptions
    |-- Prevention precautions
    |-- Severity calculation
    |
    v
[app.py] Returns JSON response to frontend
    |
    v
User sees prediction + disease info + recommendations



5. INSTALLATION & SETUP
================================================================================

PREREQUISITES:
- Python 3.8 or higher
- pip package manager
- Windows/Mac/Linux OS

STEP 1: NAVIGATE TO PROJECT DIRECTORY
Open command prompt/terminal and run:
    cd Ai_chatbot

STEP 2: INSTALL DEPENDENCIES
    pip install -r requirements.txt

Required packages:
    - Flask
    - scikit-learn
    - pandas
    - numpy

STEP 3: RUN APPLICATION
    python app.py

STEP 4: ACCESS WEB INTERFACE
Open your browser and go to:
    http://localhost:5000

The chatbot will be ready to use.



6. API DOCUMENTATION
================================================================================

ENDPOINT: POST /chat
Purpose: Conduct multi-turn medical consultation

REQUEST FORMAT:
{
  "user_id": "patient_001",
  "message": "I have a fever and cough",
  "step": 2
}

RESPONSE FORMAT (During conversation):
{
  "response": "Can you tell me how many days have you had these symptoms?",
  "step": 3,
  "prediction": null,
  "confidence": null
}

RESPONSE FORMAT (Final - with prediction):
{
  "response": "Based on your symptoms, possible diagnoses include...",
  "step": 7,
  "prediction": "Flu",
  "confidence": 0.78,
  "description": "Influenza is a contagious respiratory illness...",
  "precautions": ["Rest", "Stay hydrated", "Use tissues", "Avoid others"]
}

CONVERSATION FLOW:
Step 0: Welcome message
Step 1: Ask for patient name
Step 2: Ask for age
Step 3: Ask for gender
Step 4: Ask for symptom description
Step 5: Ask for duration of symptoms
Step 6: Ask for severity (1-10 scale)
Step 7: Return prediction with disease information



7. DATASETS
================================================================================

TRAINING DATA STRUCTURE:
- Training.csv (4,920 samples)
  Columns: disease + 131 binary symptom columns
  Values: 1 (symptom present) / 0 (symptom absent)

- Testing.csv (2,430 samples)
  Same structure for model validation

MASTER DATA:
- symptom_Description.csv
  Maps disease name → medical description

- symptom_precaution.csv
  Maps disease name → 4 preventive precautions

- Symptom_severity.csv
  Maps symptom name → severity weight (1-10 scale)



8. CONFIGURATION
================================================================================

Edit these constants in app.py to customize behavior:

MODEL_K_VALUE = 5
    Number of nearest neighbors to consider (default: 5)

CONFIDENCE_MIN = 0.35
    Minimum confidence threshold (default: 0.35 or 35%)

CONFIDENCE_MAX = 0.95
    Maximum confidence threshold (default: 0.95 or 95%)

SYMPTOM_MATCH_THRESHOLD = 0.8
    Fuzzy match similarity threshold (0-1 scale, default: 0.8)



9. MODEL DETAILS
================================================================================

ALGORITHM: K-Nearest Neighbors (KNN)
- K Value: 5 neighbors
- Distance Metric: Euclidean
- Feature Count: 131 binary symptoms

CONFIDENCE SCORING:
Confidence = (Neighbor Agreement + Symptom Coverage - Count Penalty) / 3

Where:
- Neighbor Agreement: how many of 5 neighbors predict same disease (0-5)
- Symptom Coverage: percentage of disease's typical symptoms reported (0-100%)
- Count Penalty: fewer reported symptoms = lower confidence (0-1)

SEVERITY CALCULATION:
Severity = (Sum of Symptom Weights × Duration) / (Symptom Count + 1)

Classification:
- SEVERE: Score > 13
- MODERATE: Score > 5 and ≤ 13
- MILD: Score ≤ 5



10. FILE DESCRIPTIONS
================================================================================

app.py - Flask Web Server
- Main Flask application with REST API
- Session management for multi-turn chats
- Multi-step patient intake form
- Handles /chat endpoint for conversations
- Returns JSON responses with predictions

model.py - KNN Disease Classifier
- Trains K-Nearest Neighbors model on training data
- Converts symptoms to binary feature vectors
- Predicts disease from 5 nearest neighbors
- Calculates confidence scores
- Provides accuracy metrics

preprocessing.py - Data Pipeline
- Loads training/testing datasets from CSV
- Cleans duplicate column names
- Fills missing values with 0
- Splits data 67% training / 33% testing
- Encodes disease labels for model

symptom_extractor.py - NLP Symptom Parser
Uses three extraction strategies:
1. Synonym mapping (casual language → dataset names)
   Example: "stomach ache" → "stomach_pain"
2. Exact matching (known symptoms in text)
   Example: "fever" found in "I have high fever"
3. Fuzzy matching (handles typos, 80% threshold)
   Example: "fver" matched to "fever"

utils.py - Utility Functions
- load_description(): Get disease medical descriptions
- load_severity(): Get symptom severity weights
- load_precautions(): Get disease prevention precautions
- calculate_severity(): Calculate patient severity score

test.py - Testing Script
- Tests data preprocessing correctness
- Tests model training and accuracy
- Validates predictions on test set
- Provides model performance metrics



11. TESTING
================================================================================

RUN TESTS:
    python test.py

Expected output:
- Data preprocessing verification
- Model accuracy on test set
- Sample predictions
- Performance metrics

All tests should pass without errors.



12. TROUBLESHOOTING
================================================================================

ISSUE: "Module not found" or "ImportError"
SOLUTION: Run pip install -r requirements.txt
         Make sure all packages are installed

ISSUE: "Port 5000 already in use"
SOLUTION: Change port in app.py:
         app.run(debug=True, port=5001)
         Then access http://localhost:5001

ISSUE: "CSV file not found" error
SOLUTION: Verify folder structure:
         Data/ folder with Training.csv and Testing.csv
         Master Data/ folder with CSV files
         Check file names match exactly (case-sensitive)

ISSUE: Low prediction confidence scores
REASON: Patient reported too few symptoms
SOLUTION: Ask more follow-up questions to gather details
         The model needs sufficient symptom information

ISSUE: Application runs but no web interface
SOLUTION: Check firewall settings
         Try accessing http://127.0.0.1:5000 instead
         Check browser console for errors



13. USAGE EXAMPLE
================================================================================

SAMPLE CONVERSATION:

User: "I have a fever and cough"
Bot: "Thanks for sharing. For how many days have you had these symptoms?"

User: "About 3 days"
Bot: "On a scale of 1-10, how severe are your symptoms?"

User: "7 out of 10"
Bot: "Do you have any pre-existing health conditions?"

User: "No, I'm generally healthy"
Bot: "Any family history of diseases?"

User: "My mother had pneumonia once"
Bot: "Thank you for that information. Let me analyze your symptoms..."

Bot: "Based on your symptoms, likely diagnosis: FLU (78% confidence)

Disease: Influenza
Description: Influenza (flu) is a contagious respiratory illness caused by 
influenza viruses. It spreads through respiratory droplets and can cause severe 
complications in vulnerable populations.

Precautions:
1. Get adequate rest (sleep 8-10 hours daily)
2. Stay hydrated (drink water, warm fluids)
3. Use tissues to cover coughs/sneezes
4. Avoid contact with others to prevent spread"




14. VERIFICATION CHECKLIST
================================================================================

CODE QUALITY:
☐ All Python files have docstrings
☐ No redundant code (cleaned up)
☐ Error handling in place
☐ Type hints added
☐ Configuration centralized

DOCUMENTATION:
☐ README.md complete and clear
☐ API documentation present
☐ Data schema documented
☐ Test guide provided
☐ Setup guide clear

TESTING & DEPLOYMENT:
☐ test.py runs without errors
☐ All tests pass
☐ requirements.txt complete
☐ Installation follows README
☐ App starts without errors

TEAM COLLABORATION:
☐ Each member can explain their section
☐ No duplicate work
☐ Clear file ownership
☐ Easy to extend/modify




15. QUICK REFERENCE
================================================================================

START APPLICATION:
python app.py

RUN TESTS:
python test.py

INSTALL DEPENDENCIES:
pip install -r requirements.txt

ACCESS WEB INTERFACE:
http://localhost:5000

MAIN ENDPOINT:
POST http://localhost:5000/chat

KEY MODEL PARAMETERS:
- K Value: 5
- Confidence Range: 35% - 95%
- Symptom Fuzzy Match: 80%
- Training Data: 4,920 samples
- Features: 131 symptoms




17. PROJECT STATUS & NOTES
================================================================================

Project Version: 1.0
Status: Documentation & Code Quality Focus
Focus Areas:
- Comprehensive documentation
- Code quality and cleanup
- Error handling
- Testing framework
- Deployment readiness

Next Steps :
1. Gather team feedback
2. Deploy to production
3. Monitor user experience
4. Plan Phase 2 improvements


======================================================
                            END OF DOCUMENTATION
================================================================================

