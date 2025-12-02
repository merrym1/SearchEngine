## Quick Start (Windows / macOS / Linux)

1) **Install Python 3.10+** and **Git**.
2) Open **VS Code** → `File → Open Folder...` → select this folder.
3) Open Data → Processed → hasil_newfinal.. → perform embedding using LaBse in Google Collab
```
!pip install -q sentence-transformers

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

df = pd.read_csv("/content/hasil_newfinalparsing_progress3_allhal.csv")   
texts = df["teks"].astype(str).tolist()   

model = SentenceTransformer("sentence-transformers/LaBSE")

batch_size = 64   
all_embeddings = []

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batch"):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)

embeddings = np.vstack(all_embeddings)

np.save("embeddings_alkitabbatak.npy", embeddings)

print(f"✅ Embedding selesai! Bentuk vektor: {embeddings.shape}")
```
4) Create & activate virtual env:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
5) Install dependencies:
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
