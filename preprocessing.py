import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def resolve_base_dir():

    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent]
    for candidate in candidates:
        if (candidate / "Data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the 'Data' folder. Please make sure Data/ exists."
    )


def generate_training_testing_from_raw(BASE_DIR, test_size=0.33, random_state=42):
    raw_path = BASE_DIR / "Data" / "dataset.csv"
    if not raw_path.exists():
        return

    raw_df = pd.read_csv(raw_path, header=None)
    raw_df = raw_df.astype(str).apply(lambda col: col.str.strip())
    raw_df = raw_df.replace({"nan": ""})

    records = []
    symptoms = set()
    for _, row in raw_df.iterrows():
        prognosis = row.iloc[0]
        if not prognosis:
            continue

        row_symptoms = []
        for cell in row.iloc[1:]:
            cell_str = str(cell).strip()
            if cell_str and cell_str.lower() != "nan":
                symptom = cell_str
                row_symptoms.append(symptom)
                symptoms.add(symptom)

        if not row_symptoms:
            continue

        records.append({"prognosis": prognosis.strip(), "symptoms": row_symptoms})

    if not records:
        return

    symptoms = sorted(symptoms)
    dataset = pd.DataFrame(
        [
            {symptom: int(symptom in record["symptoms"]) for symptom in symptoms}
            | {"prognosis": record["prognosis"]}
            for record in records
        ],
        columns=symptoms + ["prognosis"],
    )

    try:
        training_df, testing_df = train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state,
            stratify=dataset["prognosis"],
        )
    except ValueError:
        training_df, testing_df = train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state,
        )

    training_df.to_csv(BASE_DIR / "Data" / "Training.csv", index=False)
    testing_df.to_csv(BASE_DIR / "Data" / "Testing.csv", index=False)


def load_and_preprocess():
    BASE_DIR = resolve_base_dir()
    generate_training_testing_from_raw(BASE_DIR)

    # Load raw CSVs
    training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")
    testing  = pd.read_csv(BASE_DIR / "Data" / "Testing.csv")

    # Step 1: Clean duplicate column names e.g. "fever.1" -> "fever"
    training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
    testing.columns  = testing.columns.str.replace(r"\.\d+$", "", regex=True)

    # Step 2: Remove duplicate columns
    training = training.loc[:, ~training.columns.duplicated()]
    testing  = testing.loc[:,  ~testing.columns.duplicated()]

    # Step 3: Fill missing values with 0 (absent symptom = 0)
    training.fillna(0, inplace=True)
    testing.fillna(0, inplace=True)

    # Step 4: Separate features and target label from created Training/Testing files
    cols = training.columns[:-1]
    x_train = training[cols]
    y_train = training['prognosis']
    x_test = testing[cols]
    y_test = testing['prognosis']

    # Step 5: Encode disease names to numbers
    # e.g. "Diabetes" -> 4, "Heart attack" -> 12
    le = preprocessing.LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    print("\nDataset Preprocessing Complete:")
    print(f"   Training samples : {len(x_train)}")
    print(f"   Testing samples  : {len(x_test)}")
    print(f"   Total features   : {len(cols)} symptoms")
    print(f"   Total diseases   : {len(le.classes_)}")

    return x_train, x_test, y_train_encoded, y_test_encoded, le, cols, training, BASE_DIR


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, le, cols, training, BASE_DIR = load_and_preprocess()

    print("\nQuick data check:")
    print(f"  Base directory   : {BASE_DIR}")
    print(f"  Training shape   : {training.shape}")
    print(f"  Test split shape : {x_test.shape}")
    print(f"  Feature count    : {len(cols)}")
    print(f"  Diseases found   : {len(le.classes_)}")

    print("\nFirst training example:")
    print(training.iloc[0].to_dict())

    print("\nLabel encoding mapping:")
    for encoded_label, disease_name in enumerate(le.classes_):
        print(f"  {encoded_label} -> {disease_name}")