"""Train the random-forest classifier for the simulated electronic nose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sensor import CHANNELS, sniff

ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = ROOT / "samples.json"
MODEL_PATH = ROOT / "artifacts/nose_model.joblib"


def load_samples(path: Path = SAMPLES_PATH) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        samples = json.load(file)
    if not isinstance(samples, dict) or not samples:
        raise ValueError(f"No smell samples found in {path}")
    return samples


def train_nose(
    samples_path: Path = SAMPLES_PATH,
    model_path: Path = MODEL_PATH,
    repetitions: int = 200,
) -> float:
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    samples = load_samples(samples_path)
    rng = np.random.default_rng(42)
    features: list[np.ndarray] = []
    labels: list[str] = []

    for smell, sample in samples.items():
        molecules = sample.get("molecules")
        if not isinstance(molecules, list) or not molecules:
            raise ValueError(f"Sample {smell!r} must contain a non-empty molecules list.")
        for _ in range(repetitions):
            features.append(sniff(molecules, rng=rng))
            labels.append(smell)

    x_train, x_test, y_train, y_test = train_test_split(
        np.stack(features),
        np.asarray(labels),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(x_test)))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes": list(model.classes_),
            "channels": list(CHANNELS),
            "validation_accuracy": accuracy,
            "repetitions_per_smell": repetitions,
        },
        model_path,
    )
    print(f"Trained on {len(features):,} simulated sniffs across {len(samples)} smells.")
    print(f"Held-out simulator accuracy: {accuracy:.1%}")
    print("This score measures performance on the simulator, not on a physical sensor.")
    print(f"Saved nose model: {model_path}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    train_nose(args.samples, args.model, args.repetitions)
