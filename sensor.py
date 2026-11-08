"""An explicitly simulated eight-channel gas-sensor array."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

CHANNELS = (
    "Broad response",
    "Volatile compounds",
    "Oxygen-sensitive",
    "Aromatic compounds",
    "Sulfur-sensitive",
    "Polar compounds",
    "Oil-loving compounds",
    "Large molecules",
)


def _molecule_features(smiles: str) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    atom_count = max(molecule.GetNumHeavyAtoms(), 1)
    molecular_weight = Descriptors.MolWt(molecule)
    log_p = Descriptors.MolLogP(molecule)
    oxygen_fraction = sum(atom.GetSymbol() == "O" for atom in molecule.GetAtoms()) / atom_count
    sulfur_fraction = sum(atom.GetSymbol() == "S" for atom in molecule.GetAtoms()) / atom_count
    aromatic_fraction = rdMolDescriptors.CalcNumAromaticRings(molecule) / atom_count
    polar_groups = (Lipinski.NumHAcceptors(molecule) + Lipinski.NumHDonors(molecule)) / atom_count
    volatility = 1.0 / (1.0 + np.exp(np.clip((molecular_weight - 160.0) / 35.0, -30, 30)))
    size = 1.0 / (1.0 + np.exp(np.clip(-(molecular_weight - 90.0) / 45.0, -30, 30)))
    lipophilicity = 1.0 / (1.0 + np.exp(np.clip(-log_p, -30, 30)))

    return np.asarray(
        [
            0.4 * volatility + 0.3 * lipophilicity + 0.3 * size,
            volatility,
            min(oxygen_fraction * 2.5, 1.0),
            min(aromatic_fraction * 2.5, 1.0),
            min(sulfur_fraction * 5.0, 1.0),
            min(polar_groups * 1.4, 1.0),
            lipophilicity,
            size,
        ],
        dtype=float,
    )


def sniff(
    molecules: Sequence[str | Mapping[str, object]],
    rng: np.random.Generator | None = None,
    humidity: float | None = None,
) -> np.ndarray:
    """Simulate one sensor cycle from a list of molecule SMILES or molecule records."""
    if not molecules:
        raise ValueError("At least one molecule is required for a sniff.")
    if humidity is not None and not 0.0 <= humidity <= 1.0:
        raise ValueError("humidity must be between 0 and 1")

    generator = rng if rng is not None else np.random.default_rng()
    humidity_level = float(generator.uniform(0.25, 0.8) if humidity is None else humidity)
    smiles_list: list[str] = []
    for item in molecules:
        smiles = item.get("smiles") if isinstance(item, Mapping) else item
        if not isinstance(smiles, str):
            raise ValueError("Each molecule must provide a SMILES string.")
        smiles_list.append(smiles)

    responses = np.stack([_molecule_features(smiles) for smiles in smiles_list])
    mixture_response = responses.mean(axis=0)
    humidity_effect = humidity_level * np.asarray([0.04, 0.03, 0.12, 0.02, 0.01, 0.1, 0.02, 0.01])
    noise = generator.normal(0.0, 0.012, size=len(CHANNELS))
    return np.clip(mixture_response + humidity_effect + noise, 0.0, 1.0)
