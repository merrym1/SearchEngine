## Quick Start (Windows / macOS / Linux)

1) **Install Python 3.10+** and **Git**.
2) Open **VS Code** → `File → Open Folder...` → select this folder.
3) Create & activate virtual env:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
4) Install dependencies:
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```


## Run the App (Streamlit)

```bash
streamlit run src/app/streamlit_app_pinecone.py
```

## Project Layout

```
batak-bible-semantic-search/
├─ data/processed
│  └─ hasil_newfinalparsing_progress3_allhal.csv
├─ embeddings/embeddings_alkitabbatak.npy
├─ src/
│  ├─ app/streamlit_app.py
│  ├─ models/translate_nllb.py
│  ├─ models/embed_labse.py
│  ├─ preprocess/pdf_to_csv_lib.py
│  ├─ search/vector_store.py
│  ├─ search/search_engine.py
│  └─ utils/regex_patterns.py
├─ .env
├─ requirements.txt
└─ README.md
```

### Notes
- Default translator: `kepinsam/ind-to-bbc-nmt-v5` in `.env`.
- LaBSE: `sentence-transformers/LaBSE` (auto-downloads on first run).
