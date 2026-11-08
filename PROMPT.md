# Build prompt

Paste this into a coding agent in an empty project folder to recreate the
lightweight demo described in this repository:

> Build a Python 3.10+ project called AI Nose. It predicts odor words from
> molecular structures and demonstrates how that model can be paired with an
> electronic nose. The electronic nose must be explicitly simulated in code;
> the app must run on CPU without physical hardware.
>
> Use the OpenPOM 4,983-molecule odor dataset if it is already present locally.
> Train a CPU-friendly scikit-learn multilabel classifier on Morgan fingerprints
> generated from RDKit SMILES. Persist the trained model and print an honest
> held-out ROC-AUC metric. Do not claim a result until it is measured.
>
> Include six samples (coffee, vanilla candle, mint, lemon, banana, and wood
> smoke), each with representative molecule names and valid SMILES. Implement
> an eight-channel simulated gas sensor whose responses depend on molecule
> properties and include random measurement noise and humidity effects. Generate
> repeated sensor readings for each sample, train a random forest to classify
> the samples, and report accuracy on a held-out split from the same simulator.
> Clearly state that this score is not a measure of real hardware performance.
>
> Create a Streamlit app where a user chooses a smell and presses Sniff. Display
> the simulated channel readings, predicted sample and confidence, and the top
> odor words predicted by applying the molecule model to the known molecules
> for that predicted sample. Explain that sensor readings do not determine
> chemical composition.
>
> Include requirements and pyproject metadata, a README with setup steps and
> hardware-replacement guidance, a PROMPT.md, and offline smoke tests. Keep
> artifacts out of version control, validate inputs, show actionable model-file
> errors in the app, and avoid GPU-only or network-only requirements.
