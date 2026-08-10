# Local setup

Requires **Python 3.11 exactly**, ~6 GB free disk, macOS or Linux (Windows: use WSL2).
No API keys or accounts needed.

## 1. System dependencies

**macOS:**

```bash
brew install python@3.11 poppler tesseract openjdk
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"   # add to ~/.zshrc; /usr/bin/java is a non-functional stub
```

**Ubuntu / Debian / WSL2:**

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv poppler-utils tesseract-ocr tesseract-ocr-eng default-jre libgl1
```

Verify — each must print a version:

```bash
pdftoppm -v && tesseract --version && java -version
```

## 2. Install

```bash
git clone https://github.com/joki101502/patent-sar-miner.git
cd patent-sar-miner
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps MolScribe    # --no-deps is required: its torch<2.0 pin no longer resolves
pip install -e .
sarmine slim-checkpoint            # one-time: downloads ~1.1 GB model, writes models/molscribe_slim.pth (~384 MB)
```

Optional sanity check:

```bash
pytest    # fast hermetic suite, no network needed
```

## 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. From there everything happens in the app:
upload a patent PDF on the **Ingest** screen and run the extraction (~4–5 min
for a full patent), then browse the SAR table, shortlist, and review queue.
Corrections live in the browser session only — **export before closing the tab**.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install -e .` rejects your Python version | Recreate the venv with `python3.11 -m venv .venv` |
| py2opsin / name parsing fails on macOS | You're on the `/usr/bin/java` stub — apply the PATH fix in Step 1 |
| `pip install MolScribe` without `--no-deps` won't resolve | Expected; install exactly as in Step 2 |
| torch pulls gigabytes of `nvidia-*` packages | Install torch only via `requirements.txt` (CPU index) |
| Extraction fails mentioning patents.google.com | Tick **Force PDF path** on the Ingest screen to skip the network path |
