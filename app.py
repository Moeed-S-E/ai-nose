"""Streamlit demo combining the molecule model and simulated nose."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st

from sensor import CHANNELS, sniff
from smell_model import MODEL_PATH, load_model, predict_molecule_odors, train_model
from train_nose import MODEL_PATH as NOSE_MODEL_PATH
from train_nose import train_nose

try:
    import plotly.graph_objects as go
except ImportError:  # falls back to a bar chart
    go = None

ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = ROOT / "samples.json"

# Palette: dusk plum base, rose for the signal, mint for the readout.
INK = "#1A1030"
PLUM = "#281A45"
LILAC = "#B9A8E0"
ROSE = "#FF7AA8"
MINT = "#8FE3CF"
MIST = "#F1ECFA"


@st.cache_data
def load_samples() -> dict[str, dict[str, Any]]:
    with SAMPLES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_smell_model() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        train_model(model_path=MODEL_PATH)
    return load_model(MODEL_PATH)


@st.cache_resource
def load_nose_model() -> dict[str, Any]:
    if not NOSE_MODEL_PATH.is_file():
        train_nose(model_path=NOSE_MODEL_PATH)
    return joblib.load(NOSE_MODEL_PATH)


def esc(value: Any) -> str:
    return html.escape(str(value))


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;600;800&family=DM+Sans:wght@400;500;700&display=swap');

        html, body, [class*="css"], .stApp {{ font-family: 'DM Sans', sans-serif; }}
        .stApp {{
            background:
                radial-gradient(900px 500px at 85% -10%, rgba(255,122,168,.22), transparent 60%),
                radial-gradient(700px 500px at -10% 20%, rgba(143,227,207,.12), transparent 60%),
                {INK};
            color: {MIST};
        }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        .block-container {{ max-width: 760px; padding-top: 2.5rem; padding-bottom: 4rem; }}

        h1, h2, h3, .display {{
            font-family: 'Bricolage Grotesque', sans-serif !important;
            letter-spacing: -0.02em;
            color: {MIST};
        }}
        .hero-title {{
            font-family: 'Bricolage Grotesque', sans-serif;
            font-weight: 800;
            font-size: clamp(2.6rem, 9vw, 4.4rem);
            line-height: .95;
            letter-spacing: -0.04em;
            margin: 0 0 .6rem 0;
            background: linear-gradient(100deg, {MIST} 20%, {ROSE} 75%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }}
        .hero-sub {{ color: {LILAC}; font-size: 1.05rem; max-width: 34rem; margin-bottom: 1.4rem; }}

        .sim-note {{
            display: inline-flex; gap: .6rem; align-items: center;
            padding: .45rem .9rem; border-radius: 999px;
            border: 1px solid rgba(185,168,224,.35);
            background: rgba(185,168,224,.08);
            color: {LILAC}; font-size: .85rem; margin-bottom: 1.8rem;
        }}
        .sim-note::before {{
            content: ""; width: .5rem; height: .5rem; border-radius: 50%;
            background: {MINT}; box-shadow: 0 0 10px {MINT};
        }}

        .sample-card {{
            border-radius: 22px; padding: 1.4rem 1.5rem;
            background: linear-gradient(160deg, {PLUM}, rgba(40,26,69,.55));
            border: 1px solid rgba(185,168,224,.18);
            margin: 1rem 0 1.2rem 0;
        }}
        .sample-card .name {{
            font-family: 'Bricolage Grotesque', sans-serif;
            font-size: 1.7rem; font-weight: 600; margin-bottom: .3rem;
        }}
        .sample-card .desc {{ color: {LILAC}; line-height: 1.55; }}
        .mol {{
            display: inline-block; margin: .9rem .4rem 0 0; padding: .25rem .7rem;
            border-radius: 8px; font-size: .85rem;
            background: rgba(143,227,207,.1); color: {MINT};
            border: 1px solid rgba(143,227,207,.25);
        }}

        div.stButton > button {{
            width: 100%; border: 0; border-radius: 16px; padding: .9rem 1rem;
            font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.2rem; font-weight: 600;
            color: {INK};
            background: linear-gradient(100deg, {ROSE}, #FFB38A);
            box-shadow: 0 10px 30px -10px rgba(255,122,168,.7);
            transition: transform .15s ease, box-shadow .15s ease;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 36px -10px rgba(255,122,168,.85);
            color: {INK};
        }}
        div.stButton > button:focus-visible {{ outline: 3px solid {MINT}; outline-offset: 3px; }}

        .result {{
            border-radius: 26px; padding: 1.8rem 1.6rem; margin: 1.6rem 0 1rem 0;
            background: radial-gradient(500px 220px at 100% 0%, rgba(255,122,168,.25), transparent 70%), {PLUM};
            border: 1px solid rgba(255,122,168,.35);
        }}
        .result .label {{ color: {LILAC}; font-size: .9rem; }}
        .result .answer {{
            font-family: 'Bricolage Grotesque', sans-serif;
            font-weight: 800; font-size: clamp(2rem, 7vw, 3.2rem);
            letter-spacing: -0.03em; line-height: 1.05; margin: .2rem 0 1rem 0;
        }}
        .meter {{ height: 10px; border-radius: 99px; background: rgba(241,236,250,.12); overflow: hidden; }}
        .meter > span {{ display: block; height: 100%; border-radius: 99px;
            background: linear-gradient(90deg, {MINT}, {ROSE}); }}
        .meter-cap {{ display: flex; justify-content: space-between; margin-top: .45rem;
            color: {LILAC}; font-size: .85rem; }}

        .section-title {{
            font-family: 'Bricolage Grotesque', sans-serif;
            font-size: 1.35rem; font-weight: 600; margin: 2rem 0 .2rem 0;
        }}
        .section-note {{ color: {LILAC}; font-size: .9rem; margin-bottom: .6rem; }}

        .odor-row {{ display: grid; grid-template-columns: 8rem 1fr 3rem; gap: .8rem;
            align-items: center; margin: .5rem 0; }}
        .odor-row .w {{ font-weight: 500; text-transform: capitalize; }}
        .odor-row .bar {{ height: 8px; border-radius: 99px; background: rgba(241,236,250,.1); overflow: hidden; }}
        .odor-row .bar > span {{ display: block; height: 100%; border-radius: 99px; background: {MINT}; }}
        .odor-row .s {{ color: {LILAC}; font-size: .85rem; text-align: right; }}

        div[data-testid="stExpander"] {{
            border: 1px solid rgba(185,168,224,.2); border-radius: 16px; background: rgba(40,26,69,.4);
        }}
        div[data-baseweb="select"] > div {{
            background: {PLUM}; border-color: rgba(185,168,224,.3); border-radius: 14px;
        }}
        button[kind="pills"], button[kind="pillsActive"] {{ border-radius: 999px; }}

        @media (prefers-reduced-motion: reduce) {{
            div.stButton > button {{ transition: none; }}
        }}
        @media (max-width: 520px) {{
            .odor-row {{ grid-template-columns: 6rem 1fr 2.6rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sensor_chart(readings: np.ndarray, chart_type: str) -> None:
    """Render the sensor response as a radar or bar chart."""
    if go is None:
        import pandas as pd

        st.bar_chart(pd.DataFrame({"Response": readings}, index=CHANNELS))
        return
    values = [float(v) for v in readings]
    if chart_type == "Radar":
        labels = list(CHANNELS)
        chart = go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            line=dict(color=ROSE, width=3),
            fillcolor="rgba(255,122,168,.28)",
            marker=dict(size=7, color=MINT),
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
        fig = go.Figure(chart)
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    range=[0, 1],
                    gridcolor="rgba(185,168,224,.22)",
                    linecolor="rgba(185,168,224,.22)",
                    tickfont=dict(color=LILAC, size=10),
                ),
                angularaxis=dict(
                    gridcolor="rgba(185,168,224,.22)",
                    tickfont=dict(color=MIST, size=13, family="DM Sans"),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MIST),
            margin=dict(l=40, r=40, t=20, b=20),
            height=380,
            showlegend=False,
        )
    else:
        fig = go.Figure(
            go.Bar(
                x=list(CHANNELS),
                y=values,
                marker_color=[MINT if value < 0.5 else ROSE for value in values],
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MIST),
            yaxis=dict(range=[0, 1], title="Response"),
            xaxis=dict(title=None, tickangle=-25),
            margin=dict(l=30, r=20, t=20, b=90),
            height=380,
            showlegend=False,
        )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def parse_custom_molecules(text: str) -> list[dict[str, str]]:
    """Parse one SMILES per line, optionally in `name | SMILES` form."""
    molecules: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            name, smiles = (part.strip() for part in line.split("|", maxsplit=1))
            if not name or not smiles:
                raise ValueError(f"Line {line_number}: use `name | SMILES`.")
        else:
            name, smiles = f"Molecule {len(molecules) + 1}", line
        molecules.append({"name": name, "smiles": smiles})
    if not molecules:
        raise ValueError("Enter at least one SMILES string.")
    return molecules


def describe_molecules(
    molecules: list[dict[str, str]], model: dict[str, Any], top_n: int
) -> list[tuple[str, float]]:
    """Average molecule odor scores into one ranked mixture description."""
    predictions = [
        predict_molecule_odors(molecule["smiles"], model, top_n=len(model["odor_labels"]))
        for molecule in molecules
    ]
    mean_scores = {
        odor: float(np.mean([dict(result)[odor] for result in predictions]))
        for odor in model["odor_labels"]
    }
    return sorted(mean_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def odor_bars(top_odors: list[tuple[str, float]]) -> str:
    peak = max((score for _, score in top_odors), default=1.0) or 1.0
    rows = "".join(
        f'<div class="odor-row"><div class="w">{esc(odor)}</div>'
        f'<div class="bar"><span style="width:{max(score / peak, 0.03):.0%}"></span></div>'
        f'<div class="s">{score:.0%}</div></div>'
        for odor, score in top_odors
    )
    return rows


st.set_page_config(page_title="AI Nose", page_icon="👃", layout="centered")
inject_css()

st.markdown('<div class="hero-title">AI Nose</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Pick a smell. A simulated eight-channel gas sensor sniffs it, '
    "then a molecule model puts words to what it found.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sim-note">Simulated sensor: scores are not measurements from physical hardware.</div>',
    unsafe_allow_html=True,
)

missing_models = []
if not NOSE_MODEL_PATH.is_file():
    missing_models.append("simulated nose")
if not MODEL_PATH.is_file():
    missing_models.append("molecule odor")
if missing_models:
    st.info(
        f"First use will train the missing {', '.join(missing_models)} model(s) "
        "on this app instance. The first Sniff may take a few minutes."
    )

samples = load_samples()
with st.sidebar:
    st.header("Sniff controls")
    input_mode = st.radio("Sample source", ["Demo sample", "Custom SMILES"])
    humidity_percent = st.slider(
        "Simulated humidity", min_value=25, max_value=80, value=50, format="%d%%"
    )
    repeat_count = st.slider("Repeated sniffs to average", min_value=1, max_value=10, value=3)
    odor_count = st.slider("Odor words to show", min_value=3, max_value=10, value=5)
    chart_type = st.selectbox("Sensor chart", ["Radar", "Bar"])
    show_probabilities = st.checkbox("Show all sample probabilities", value=True)
    show_molecule_scores = st.checkbox("Show molecule-by-molecule odor scores", value=False)

if input_mode == "Demo sample":
    sample_names = list(samples)
    if hasattr(st, "pills"):
        choice = st.pills(
            "Choose a smell",
            sample_names,
            default=sample_names[0],
            selection_mode="single",
        )
        selected_smell = choice or sample_names[0]
    else:
        selected_smell = st.selectbox("Choose a smell", sample_names)
    selected_sample = samples[selected_smell]
    molecules = selected_sample["molecules"]
    sample_description = selected_sample["description"]
else:
    selected_smell = "Custom sample"
    sample_description = "Enter one molecule SMILES per line, or use `name | SMILES`."
    custom_smiles = st.text_area(
        "Molecule SMILES",
        placeholder="Vanillin | COc1cc(C=O)ccc1O\nCC(=O)OCCC(C)C",
        help="Each non-empty line is a SMILES. Optionally add a display name before `|`.",
        height=140,
    )
    molecules = []

    try:
        molecules = parse_custom_molecules(custom_smiles)
    except ValueError:
        pass

if molecules:
    molecule_chips = "".join(
        f'<span class="mol">{esc(molecule["name"])}</span>' for molecule in molecules
    )
    st.markdown(
        f'<div class="sample-card"><div class="name">{esc(selected_smell)}</div>'
        f'<div class="desc">{esc(sample_description)}</div>{molecule_chips}</div>',
        unsafe_allow_html=True,
    )
elif input_mode == "Custom SMILES":
    st.info("Enter one or more SMILES above, then press Sniff.")

if st.button("Sniff", type="primary", disabled=not molecules):
    try:
        with st.spinner("Sniffing..."):
            nose_data = load_nose_model()
            smell_data = load_smell_model()
            rng = np.random.default_rng()
            repeated_readings = np.stack(
                [
                    sniff(molecules, rng=rng, humidity=humidity_percent / 100)
                    for _ in range(repeat_count)
                ]
            )
            readings = repeated_readings.mean(axis=0)
            nose_model = nose_data["model"]
            probabilities = nose_model.predict_proba(readings.reshape(1, -1))[0]
            prediction_index = int(np.argmax(probabilities))
            predicted_smell = str(nose_model.classes_[prediction_index])
            confidence = float(probabilities[prediction_index])

            odor_molecules = (
                samples[predicted_smell]["molecules"]
                if input_mode == "Demo sample"
                else molecules
            )
            top_odors = describe_molecules(odor_molecules, smell_data, odor_count)
            molecule_predictions = (
                {
                    molecule["name"]: predict_molecule_odors(
                        molecule["smiles"], smell_data, top_n=odor_count
                    )
                    for molecule in odor_molecules
                }
                if show_molecule_scores
                else {}
            )
            class_probabilities = sorted(
                zip(nose_model.classes_, probabilities, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )

        st.markdown(
            f'<div class="result"><div class="label">The nose thinks this is</div>'
            f'<div class="answer">{esc(predicted_smell)}</div>'
            f'<div class="meter"><span style="width:{confidence:.1%}"></span></div>'
            f'<div class="meter-cap"><span>Confidence</span><span>{confidence:.1%}</span></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Sensor response</div>'
            f'<div class="section-note">Mean response across {repeat_count} simulated sniff(s); '
            f'humidity setting {humidity_percent}%.</div>',
            unsafe_allow_html=True,
        )
        sensor_chart(readings, chart_type)

        if show_probabilities:
            st.markdown(
                '<div class="section-title">Other sample matches</div>'
                '<div class="section-note">Random-forest vote probabilities for the six trained labels.</div>',
                unsafe_allow_html=True,
            )
            for class_name, probability in class_probabilities:
                st.write(f"{class_name}: {probability:.1%}")

        st.markdown(
            '<div class="section-title">What it smells like</div>'
            '<div class="section-note">Average molecule-level model scores; these are not '
            'measured concentrations or human confidence ratings.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(odor_bars(top_odors), unsafe_allow_html=True)
        if show_molecule_scores:
            with st.expander("Molecule-by-molecule odor scores"):
                for molecule_name, predictions in molecule_predictions.items():
                    st.write(f"**{molecule_name}**")
                    st.write(", ".join(f"{odor} ({score:.0%})" for odor, score in predictions))

        result = {
            "input_mode": input_mode,
            "input_sample": selected_smell,
            "molecules": molecules,
            "simulated_humidity_percent": humidity_percent,
            "sniffs_averaged": repeat_count,
            "sensor_chart": chart_type,
            "sensor_channel_means": {
                channel: float(value) for channel, value in zip(CHANNELS, readings, strict=True)
            },
            "predicted_demo_sample": predicted_smell,
            "predicted_demo_sample_probability": confidence,
            "demo_sample_probabilities": {
                str(name): float(probability) for name, probability in class_probabilities
            },
            "top_odor_model_scores": [
                {"odor": odor, "score": score} for odor, score in top_odors
            ],
            "warning": "Sensor readings are simulated, not physical measurements.",
        }
        st.download_button(
            "Download this sniff as JSON",
            data=json.dumps(result, indent=2),
            file_name="ai_nose_result.json",
            mime="application/json",
        )
    except FileNotFoundError as error:
        st.error(str(error))
    except ValueError as error:
        st.error(f"Could not process this sample: {error}")

with st.expander("How it works — detailed walkthrough"):
    st.markdown(
        """
        **1. Choose or enter molecules.** Demo recipes contain representative
        molecules written as SMILES strings, a compact text notation for
        molecular structure. In custom mode, add one SMILES per line. A line
        may optionally be written as `name | SMILES`.

        **2. Simulate the sensor.** RDKit parses each structure and computes
        simple properties such as molecular weight, oxygen/sulfur content,
        aromatic rings, polarity, and lipophilicity. The simulator combines
        those into eight normalized channel responses, adds humidity within
        the 25–80% range used to train the classifier, then adds small random
        noise to each channel. These are illustrative software values, not
        real gas-sensor physics or hardware readings.

        **3. Average repeat sniffs.** The app repeats that simulation using
        independent noise draws and averages each channel. This makes the
        display less noisy; it does not add real-world measurement evidence.

        **4. Classify the sample.** A random forest trained on simulated
        readings outputs probabilities for the six demo recipe names. A custom
        mixture is therefore matched to the closest learned demo class; it is
        not identifying arbitrary substances from a physical sensor. The
        displayed confidence is a model probability, not a calibrated guarantee.

        **5. Predict odor words from structure.** Separately, each molecule is
        converted to a 1,024-bit Morgan fingerprint (radius 2). A multilabel
        random forest trained on 4,983 OpenPOM molecules predicts scores for
        138 odor descriptors. In demo mode, the known molecules for the
        predicted sample are described. In custom mode, the entered molecules
        are described. Mixture scores are arithmetic averages of those
        molecule-level model scores.

        **Important limitation:** the sensor classifier does not discover the
        sample's chemical composition. The custom SMILES are supplied directly
        to the molecule model; the simulated nose only chooses among its six
        trained example labels. Simulator validation accuracy is not a measure
        of performance on a real sensor.
        """
    )