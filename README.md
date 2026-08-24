# Multi-Disease Detection Framework — Deep Neural Networks & Multimodal AI

A hybrid clinical-decision-**support** framework that pairs your trained **custom CNN**
(94.82% validation accuracy across **30 disease classes**) with an **LLM reasoning layer**.
The CNN is the *perception* engine; **Claude** is the *reasoning* engine, enriched by three
**contextual-memory sources** — a structured Knowledge Base, **Microsoft BioGPT**, and
**Med-PaLM 2** — to turn a raw softmax vector into a calibrated, source-attributed,
safety-bounded interpretation.

It ships as an **interactive web application** you can actually test in the browser:
drag in a medical image → get the CNN prediction, a **Grad-CAM** attention heatmap, the
LLM's contextual reasoning, a **prediction history log**, and a one-click **PDF clinical report**.

> ⚠️ **Research / educational prototype — not a medical device.** Every output is decision-support, never a diagnosis.

---

## What's inside

```
multi-disease-detection-app/
├── app/
│   ├── main.py                 # FastAPI app: /api/health, /api/analyze, /api/report, serves the UI
│   ├── config.py               # settings from .env (pydantic-settings)
│   ├── schemas.py              # pydantic request/response models
│   ├── ml/
│   │   ├── cnn_model.py        # your Cell-10 architecture, weight loading, demo mode, predict()
│   │   └── gradcam.py          # Grad-CAM heatmap + plain-English "where the model looked"
│   ├── llm/
│   │   ├── prompts.py          # the prompt engineering (system + synthesis prompt)
│   │   ├── claude_client.py    # Anthropic call + robust JSON extraction
│   │   ├── disease_metadata.py # structured Knowledge Base for all 30 classes  [KB]
│   │   ├── biogpt_adapter.py   # Microsoft BioGPT source (real via HF, else curated fallback) [BioGPT]
│   │   ├── medpalm_adapter.py  # Med-PaLM 2 source (labeled placeholder; Vertex MedLM path documented) [Med-PaLM]
│   │   └── orchestrator.py     # coordinates the 3 memory sources + Claude, with rule-based fallback
│   └── report/
│       └── pdf_report.py       # clinical PDF export (reportlab)
├── frontend/                   # vanilla HTML/CSS/JS "diagnostic console" (no build step)
├── models/
         └──best_model.weights.h5
         └──readme.md                   
├── tools/
│   └── export_class_names_snippet.py   # run in your Colab to regenerate the authoritative class_names.json
├── class_names.json            # ordered label list (index -> class) — VERIFY THIS (see below)
├── pyproject.toml              # uv-managed dependencies
├── .python-version             # 3.13
└── .env.example                # copy to .env and add your key
```

---

## ⭐ 3 things I need from you (important)

The app **runs immediately in demo mode** without any of these, but for *real* results:

### 1. Add your Anthropic API key
Claude powers the reasoning layer. Copy the example env file and paste your key:

```bash
cp .env.example .env
```

Then open `.env` and set:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

Without a valid `sk-ant-...` key the app still works, but the interpretation comes from the
built-in **rule-based fallback** instead of Claude (the UI shows "claude offline").

### 2. Drop your trained weights into `models/`
Copy your file so the path matches `.env` (`CNN_WEIGHTS_PATH=models/best_model.weights.h5`):

```
models/best_model.weights.h5
```

Without it, predictions are **simulated (demo mode)** — the UI clearly flags this with a banner.

### 3. ✅ Verify `class_names.json` matches your training label order  ← most important
This is the one thing I could **not** fully confirm from the notebook. Your CNN outputs a vector of
30 probabilities; index *i* only means something if `class_names.json[i]` is the label that index
was trained on. In your notebook the order came from `sorted(df['label'].unique())`
(Python case-sensitive sort — **uppercase before lowercase**). I reconstructed that list, but a few
leaf-label spellings were guesses.

**To generate the authoritative file**, run the snippet in `tools/export_class_names_snippet.py`
inside your Colab/training notebook (where `df` exists) and replace `class_names.json`:

```python
# in your notebook, after df is built:
import json
class_names = sorted(df['label'].unique())          # EXACT training order
json.dump(class_names, open('class_names.json', 'w'), indent=2)
print(len(class_names), "classes written")
```

Download that `class_names.json` and drop it in the project root. If the order is wrong, the model
will still run but labels will be **mismatched**, so please do this before trusting any real prediction.

---

## Run it in VS Code with `uv` (step by step)

This project uses [**uv**](https://docs.astral.sh/uv/) — a fast Python package manager — **not pip**.

### Step 0 — Install uv (once, if you don't have it)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then confirm:
```bash
uv --version
```

### Step 1 — Open the project in VS Code
```bash
cd path/to/multi-disease-detection-app
code .
```
Install the Microsoft **Python** extension if prompted. (Optional: open a terminal in VS Code with `` Ctrl+` ``.)

### Step 2 — Create the environment & install dependencies
From the project folder, run:
```bash
uv sync
```
This reads `pyproject.toml`, uses **Python 3.13** (per `.python-version`), creates a `.venv/`, and
installs everything (FastAPI, **TensorFlow ≥ 2.19** — the first line that ships Python-3.13 wheels —
the Anthropic SDK, reportlab, …). First run pulls TensorFlow, so give it a few minutes.

> **Python 3.13 note:** the dependency is written as `tensorflow>=2.19,<3` (a floor, not an exact
> pin) precisely so uv resolves the newest build that has a wheel for **your** interpreter and OS.
> On Windows/macOS/Linux with Python 3.13 this picks up a `cp313` wheel automatically. If you ever
> see *"no matching distribution for tensorflow"*, your Python is newer than any published TF wheel —
> either let uv fall back to 3.12 (`uv sync --python 3.12`) or wait for the next TF release.

> **Coming from an older setup?** If you previously ran this on Python 3.11, delete the stale
> environment and lockfile first so uv re-resolves cleanly for 3.13:
> ```powershell
> # Windows PowerShell, from the project folder
> Remove-Item -Recurse -Force .venv, .venv-test, uv.lock -ErrorAction SilentlyContinue
> ```

> **Point VS Code at the venv:** `Ctrl+Shift+P` → *Python: Select Interpreter* → choose
> `./.venv/bin/python` (Windows: `.venv\Scripts\python.exe`). This makes Run/Debug use the right env.

### Step 3 — Configure your key and files
```bash
cp .env.example .env      # then edit .env and paste your ANTHROPIC_API_KEY
```
Drop `best_model.weights.h5` into `models/`, and replace `class_names.json` (see the section above).

### Step 4 — Run the server
```bash
uv run uvicorn app.main:app --reload --port 8000
```
`uv run` executes inside the project venv (no manual "activate" needed). `--reload` restarts on code changes.

### Step 5 — Open the app
Go to **http://127.0.0.1:8000** in your browser. Drop in a medical image, optionally add patient
context, and click **Analyze**. Interactive API docs are at **http://127.0.0.1:8000/docs**.

To stop the server: `Ctrl+C` in the terminal.

---

## Optional — enable the *real* BioGPT model

By default BioGPT returns curated, clearly-labeled biomedical context (fast, no download). To run
the actual **Microsoft BioGPT** locally (adds `torch` + `transformers`, ~1.5 GB on first use):

```bash
uv sync --extra biogpt
```

Then set `ENABLE_BIOGPT=true` in `.env` and restart. (Med-PaLM 2 has no public API — it stays a
labeled placeholder; the real Google **Vertex AI MedLM** wiring is sketched in `medpalm_adapter.py`.)

---

## How the reasoning pipeline works

```
   image ─► CNN (perception) ─► top-k probabilities + Grad-CAM attention
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                            ▼                           ▼
        [KB] knowledge base          [BioGPT] biomedical         [Med-PaLM] clinical-QA
        (organ, modality,             narrative context           workup context
         category, severity)                                       (placeholder)
              └───────────────────────────┼───────────────────────────┘
                                          ▼
                            Claude  (reasoning / synthesis)
                                          ▼
              calibrated, source-attributed, safety-bounded JSON interpretation
                                          ▼
                       web console  +  downloadable PDF clinical report
```

If Claude is unavailable (no key/offline), a deterministic **rule-based fallback** produces the same
JSON shape, so the UI always shows a complete result.

---

## Features you asked for

- **Grad-CAM heatmaps** — visual "where the model looked", with a heatmap/original toggle.
- **Patient context input** — age, sex, symptoms, history, notes feed the reasoning layer.
- **PDF report export** — one-click clinical-style report (prediction ladder, Grad-CAM, interpretation).
- **Prediction history log** — every analysis in the session is kept; click to revisit.

---

## API quick reference

| Method | Route           | Purpose                                                      |
|--------|-----------------|--------------------------------------------------------------|
| GET    | `/api/health`   | model + LLM source status (used by the status pills)         |
| POST   | `/api/analyze`  | multipart image (+ patient fields) → prediction, Grad-CAM, interpretation |
| POST   | `/api/report`   | analysis JSON → downloadable PDF                             |
| GET    | `/`             | the web console                                              |

---

## Troubleshooting

- **"claude offline" pill / fallback reasoning** — check `ANTHROPIC_API_KEY` in `.env` starts with `sk-ant-`.
- **"demo mode" banner** — weights not found; confirm `models/best_model.weights.h5` exists and `CNN_WEIGHTS_PATH` matches.
- **Labels look wrong** — regenerate `class_names.json` from your notebook (see step 3 / the ⭐ section).
- **`uv sync` slow the first time** — that's TensorFlow downloading; subsequent runs are cached.
- **`No matching distribution found for tensorflow` on Python 3.13** — your interpreter is newer than any published TF wheel. Fix with `uv sync --python 3.12` (uv fetches a private 3.12 just for this project; your system 3.13 is untouched), then run everything with `uv run` as normal.
- **Weights won't load / shape mismatch** — the architecture in `cnn_model.py` must match training exactly; it was copied verbatim from Cell 10. If you changed the model, update `build_custom_cnn()` to match.
- **Quick offline sanity check** — `uv run python verify_offline.py` runs a dependency-light smoke test of the metadata, prompts, adapters, orchestrator fallback, demo predictor, Grad-CAM helpers, and PDF render (49 checks, no TF/network needed).
- **Full real-pipeline check** — `uv run python tools/smoke_test.py` loads the *real* weights, runs 5 CNN predictions with Grad-CAM, and does a live Claude call. It fails loudly if it drops to demo mode. Pass image paths to use your own: `uv run python tools/smoke_test.py img1.png img2.png`.

---

*Built to extend your trained model with an LLM contextual-reasoning layer. Not for clinical use.*
