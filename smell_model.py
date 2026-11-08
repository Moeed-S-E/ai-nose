"""Train and use a CPU-friendly molecule-to-odor model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "openpom-main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
MODEL_PATH = ROOT / "artifacts/smell_model.joblib"
FINGERPRINT_BITS = 1024
FINGERPRINT_RADIUS = 2
FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
)


def smiles_to_fingerprint(smiles: str) -> np.ndarray:
    """Convert a SMILES string to a Morgan bit vector, rejecting invalid input."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    fingerprint = FINGERPRINT_GENERATOR.GetFingerprint(molecule)
    result = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


def _read_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"OpenPOM dataset not found at {path}. Keep the included "
            "openpom-main directory beside smell_model.py."
        )

    data = pd.read_csv(path)
    if "nonStereoSMILES" in data.columns:
        smiles_column = "nonStereoSMILES"
    else:
        smiles_column = str(data.columns[0])

    odor_columns = [
        column
        for column in data.columns
        if column not in {smiles_column, "descriptors"}
    ]
    if not odor_columns:
        raise ValueError(f"No odor-label columns found in {path}")

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for smiles, row in zip(data[smiles_column], data[odor_columns].to_numpy()):
        if not isinstance(smiles, str):
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        fingerprint = FINGERPRINT_GENERATOR.GetFingerprint(molecule)
        bits = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        features.append(bits)
        labels.append(np.asarray(row, dtype=np.uint8))

    if len(features) < 10:
        raise ValueError(f"Only {len(features)} valid molecules found in {path}")

    return np.stack(features), np.stack(labels), [str(label) for label in odor_columns]


def _positive_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    outputs = model.predict_proba(features)
    probabilities = np.zeros((len(features), len(outputs)), dtype=float)
    for index, output in enumerate(outputs):
        classes = model.classes_[index]
        positive = np.flatnonzero(classes == 1)
        if len(positive):
            probabilities[:, index] = output[:, positive[0]]
    return probabilities


def load_model(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained smell model not found at {model_path}. Run `python smell_model.py` first."
        )
    return joblib.load(model_path)


def predict_molecule_odors(
    smiles: str, saved_model: dict[str, Any], top_n: int = 5
) -> list[tuple[str, float]]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    features = smiles_to_fingerprint(smiles).reshape(1, -1)
    scores = _positive_probabilities(saved_model["model"], features)[0]
    best = np.argsort(scores)[::-1][:top_n]
    return [(saved_model["odor_labels"][index], float(scores[index])) for index in best]


def train_model(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> float:
    """Train the multilabel random forest and save it; return held-out macro ROC-AUC."""
    features, labels, odor_labels = _read_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    probabilities = _positive_probabilities(model, x_test)

    aucs = [
        roc_auc_score(y_test[:, index], probabilities[:, index])
        for index in range(y_test.shape[1])
        if np.unique(y_test[:, index]).size == 2
    ]
    if not aucs:
        raise ValueError("The held-out split has no odor labels with both classes.")
    mean_auc = float(np.mean(aucs))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "odor_labels": odor_labels,
            "fingerprint_bits": FINGERPRINT_BITS,
            "fingerprint_radius": FINGERPRINT_RADIUS,
            "validation_roc_auc": mean_auc,
        },
        model_path,
    )
    print(f"Trained on {len(features):,} molecules and {len(odor_labels)} odor labels.")
    print(f"Held-out macro ROC-AUC: {mean_auc:.3f}")
    print(f"Saved model: {model_path}")
    return mean_auc


def predict_odors(smiles: str, model_path: Path = MODEL_PATH, top_n: int = 5) -> list[tuple[str, float]]:
    """Return the most likely odor labels for one molecule."""
    return predict_molecule_odors(smiles, load_model(model_path), top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    train_model(args.dataset, args.model)
