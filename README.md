# Plum Claims — AI-driven health-insurance claims processor

A multi-agent claims processing system built that,
Submits → verifies → extracts → decides → traces — with full observability.

**Status:** 12 / 12 test cases passing (see `eval_report.md`).

---

## What it does 
A member uploads medical documents (PDFs / images), the system OCRs them, classifies each document, runs them through a chain of agents (intake → verifier → quality → extractor → consistency → semantic → fraud → rules → reasoner → validator), and produces an APPROVED / PARTIAL / REJECTED / MANUAL_REVIEW / BLOCKED decision with a full agent-by-agent trace. Financial math is deterministic (Python rules engine); semantic understanding (diagnosis classification, line-item categorization, patient identity, reasoning prose) is delegated to Gemini, with deterministic fallbacks for every LLM call so the pipeline never blocks on LLM failure.

---
## Deployed URL
https://plumhq-euubaucde2rww9k4xypw4x.streamlit.app/

Note: Scanned documents will not work as there is no binaries installed in streamlit cloud and also LLM API key is not configured,for such cases configure in local using the steps mentioned below 
---

## Quickstart — clone & run locally

```bash
# 1. Clone
git clone <your-repo-url> plum-claims
cd plum-claims

# 2. Create a Python 3.11 venv
py -3.11 -m venv .venv          # Windows
# OR
python3.11 -m venv .venv        # macOS / Linux

# 3. Activate
.venv\Scripts\activate          # Windows PowerShell
# OR
source .venv/bin/activate       # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env — see "Environment variables" below
```

> **API keys are NOT bundled with this repo.** You need to add your own
> Gemini API key and the path to your local Tesseract install in `.env`
> before running the app (see below).

### Environment variables (`.env`)

```
GEMINI_API_KEY=your_gemini_api_key_here
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
GEMINI_MODEL=gemini-2.5-flash-lite
DB_PATH=claims.db
UPLOAD_DIR=uploads
```

- `GEMINI_API_KEY` — get a free key at <https://aistudio.google.com/apikey>. Without one, the system still works using deterministic keyword fallbacks (eval still passes 12/12) — but you lose semantic understanding for messy real-world documents.
- `TESSERACT_CMD` — the **absolute path** to your local `tesseract.exe`. Install Tesseract first:
  - Windows: <https://github.com/UB-Mannheim/tesseract/wiki> — installs to `C:\Program Files\Tesseract-OCR\tesseract.exe` by default
  - macOS: `brew install tesseract`
  - Linux: `apt-get install tesseract-ocr`
- `GEMINI_MODEL` — defaults to `gemini-2.5-flash-lite`. 

### Run the app

```bash
# Streamlit UI on http://localhost:8501
.venv\Scripts\streamlit.exe run streamlit_app.py
```

Then in the UI:
1. **Policy Upload** — drop `policy_terms.json` to seed the DB (one-time).
2. **Claim Upload** — pick a member, fill claim details, upload medical docs, submit.
3. **History** — browse past claims with their full agent trace.

### Run the eval suite (all 12 test cases)

```bash
.venv\Scripts\python.exe eval.py
# generates eval_report.md
```

## Repository layout

```
plum/
├── streamlit_app.py          # entire UI (3 pages: Policy Upload, Claim Upload, History)
├── eval.py                   # runs all 12 test cases → eval_report.md
├── policy_terms.json         # policy + member roster (used to seed DB)
├── test_cases.json           # 12 evaluation scenarios
├── requirements.txt
├── .env.example
├── ARCHITECTURE.md           # design doc with scaling notes
├── README.md                 # this file
├── eval_report.md            # generated
├── claims.db                 # SQLite DB (created on first run, gitignored)
├── uploads/                  # uploaded files by claim id (gitignored)
│
└── services/
    ├── __init__.py           # logger setup + package init
    ├── models.py             # all Pydantic types (component contracts)
    ├── db.py                 # SQLite schema + repository
    ├── trace.py              # TraceLogger (observability backbone)
    ├── ocr.py                # pdfplumber + Tesseract router
    ├── llm.py                # Gemini wrapper with deterministic fallbacks
    ├── pipeline.py           # orchestrator
    └── agents/               # 9 specialized agents
        ├── base.py
        ├── intake.py
        ├── classifier.py
        ├── gates.py          # verifier + quality + consistency
        ├── extractor.py
        ├── semantic.py
        ├── fraud.py
        ├── rules.py
        └── decision.py       # reasoner + validator
```

---

## What's covered in the design

- Multi-agent pipeline (bonus per assignment rubric)
- Hybrid LLM + deterministic rules — LLM for semantic understanding, Python for financial math
- Full trace persisted per claim (SQLite `traces` table)
- Failure isolation per agent — TC011 (component failure) produces APPROVED with reduced confidence + manual-review flag
- Async I/O + parallel per-document OCR/extraction
- Pydantic contracts at every agent boundary
- Per-LLM-call deterministic fallback (keyword tables, RapidFuzz, templated prose)
- Multi-model Gemini fallback chain on quota exhaustion

See `ARCHITECTURE.md` for design rationale, decisions/trade-offs, and the scaling-to-10x section.
