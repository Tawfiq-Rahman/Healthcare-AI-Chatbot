import re
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore")


# Common phrase mappings to dataset symptom names
# Handles everyday language that patients might use
symptom_synonyms = {
    "stomach ache"       : "stomach_pain",
    "belly pain"         : "stomach_pain",
    "tummy pain"         : "stomach_pain",
    "loose motion"       : "diarrhea",
    "motions"            : "diarrhea",
    "high temperature"   : "fever",
    "temperature"        : "fever",
    "feaver"             : "fever",      # common typo
    "coughing"           : "cough",
    "throat pain"        : "sore_throat",
    "cold"               : "chills",
    "breathing issue"    : "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache"          : "muscle_pain",
}


def extract_symptoms(user_input, all_symptoms):
    """
    Extracts valid symptom names from a free-text user sentence.

    Uses 3 strategies in order:
    1. Synonym replacement — maps everyday phrases to dataset names
       e.g. "stomach ache" -> "stomach_pain"

    2. Exact match — checks if any known symptom appears in the text
       e.g. "I have fever" -> detects "fever"

    3. Fuzzy match — handles typos using difflib
       e.g. "fver" -> matches "fever" (cutoff: 80% similarity)

    Returns:
        list of matched symptom strings (deduplicated)
    """
    extracted = []
    text = user_input.lower().replace("-", " ")

    # Strategy 1: Synonym replacement
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    # Strategy 2: Exact match against all known symptoms
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    # Strategy 3: Fuzzy match word by word
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