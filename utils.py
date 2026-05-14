# utils.py
# Member 2 — Data Engineer
# Responsible for: loading master data dictionaries, severity calculation

import csv
import warnings
warnings.filterwarnings("ignore")


def load_description(base_dir):
    """
    Loads disease descriptions from symptom_Description.csv.
    Returns a dict: { disease_name: description_text }
    """
    description_list = {}
    with open(base_dir / "Master Data" / "symptom_Description.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                description_list[row[0]] = row[1]
    return description_list


def load_severity(base_dir):
    """
    Loads symptom severity weights from Symptom_severity.csv.
    Returns a dict: { symptom_name: severity_score (int) }
    Higher score = more serious symptom.
    """
    severity_dict = {}
    with open(base_dir / "Master Data" / "Symptom_severity.csv") as f:
        for row in csv.reader(f):
            try:
                severity_dict[row[0]] = int(row[1])
            except:
                pass
    return severity_dict


def load_precautions(base_dir):
    """
    Loads precaution recommendations from symptom_precaution.csv.
    Returns a dict: { disease_name: [prec1, prec2, prec3, prec4] }
    """
    precaution_dict = {}
    with open(base_dir / "Master Data" / "symptom_precaution.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 5:
                precaution_dict[row[0]] = [row[1], row[2], row[3], row[4]]
    return precaution_dict


def calculate_severity(symptoms_list, num_days, severity_dict):
    """
    Calculates an overall severity score for the patient.

    Formula: (sum of symptom weights × num_days) / (symptom count + 1)

    Higher score = more urgent condition.
    Thresholds:
        score > 13 or self-reported >= 8 → SEVERE
        score > 5                        → MODERATE
        else                             → MILD
    """
    total = sum(severity_dict.get(s, 0) for s in symptoms_list)
    return round((total * num_days) / (len(symptoms_list) + 1), 2)