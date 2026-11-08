"""Offline smoke tests for the molecule model and simulated sensor."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from rdkit import rdBase

from sensor import CHANNELS, sniff
from smell_model import FINGERPRINT_BITS, smiles_to_fingerprint

ROOT = Path(__file__).resolve().parent


class SmellAppTests(unittest.TestCase):
    def test_dataset_is_bundled(self) -> None:
        dataset = ROOT / "openpom-main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
        self.assertTrue(dataset.is_file())
        with dataset.open(encoding="utf-8") as file:
            self.assertIn("nonStereoSMILES", file.readline())

    def test_samples_have_valid_sensor_inputs(self) -> None:
        with (ROOT / "samples.json").open(encoding="utf-8") as file:
            samples = json.load(file)
        self.assertEqual(len(samples), 6)
        for sample in samples.values():
            readings = sniff(sample["molecules"], rng=np.random.default_rng(42), humidity=0.5)
            self.assertEqual(readings.shape, (len(CHANNELS),))
            self.assertTrue(np.all(np.isfinite(readings)))
            self.assertTrue(np.all((readings >= 0.0) & (readings <= 1.0)))

    def test_fingerprint_has_expected_size_and_rejects_invalid_smiles(self) -> None:
        fingerprint = smiles_to_fingerprint("CCO")
        self.assertEqual(fingerprint.shape, (FINGERPRINT_BITS,))
        self.assertTrue(np.isin(fingerprint, [0, 1]).all())
        with rdBase.BlockLogs():
            with self.assertRaises(ValueError):
                smiles_to_fingerprint("not a molecule")

    def test_sensor_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            sniff([])
        with self.assertRaises(ValueError):
            sniff(["CCO"], humidity=1.5)


if __name__ == "__main__":
    unittest.main()
