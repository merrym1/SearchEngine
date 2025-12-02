
# src/vecdb/pinecone_indexer.py
import os
import numpy as np
import pandas as pd
from tqdm import trange
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH", "data/processed/hasil_newfinalparsing_progress3_allhal.csv")
EMB_PATH = os.getenv("EMB_PATH", "embeddings/embeddings_alkitabbatak.npy")
INDEX_NAME = os.getenv("PINECONE_INDEX", "bible-batak-labse")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_3fY3du_CDsFipaY8jALNt9Yx8TFrf8aPo3NnBPvkdtmzt4gCLq5ngdBUxHspgEZGUa39bR")

assert API_KEY, "pcsk_3fY3du_CDsFipaY8jALNt9Yx8TFrf8aPo3NnBPvkdtmzt4gCLq5ngdBUxHspgEZGUa39bR"

pc = Pinecone(api_key=API_KEY)

# buat index kalau belum ada
existing = [x["name"] for x in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,     # LaBSE = 768
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION),
    )
index = pc.Index(INDEX_NAME)

df = pd.read_csv(CSV_PATH)
E = np.load(EMB_PATH).astype("float32")
assert len(df) == E.shape[0], f"CSV({len(df)}) != NPY({E.shape[0]})"


E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


BATCH = 200
for start in trange(0, len(df), BATCH, desc="Upserting"):
    vectors = []
    for i in range(start, min(start + BATCH, len(df))):
        row = df.iloc[i]
        meta = {
            "kitab": str(row.get("kitab", "")),
            "pasal": int(row.get("pasal", "")),
            "ayat": int(row.get("ayat", "")),
            "teks": str(row.get("teks", "")),
        }
        vectors.append({"id": str(i), "values": E[i].tolist(), "metadata": meta})
    if vectors:
        index.upsert(vectors=vectors)

print("✅ Selesai upsert ke Pinecone:", INDEX_NAME)
