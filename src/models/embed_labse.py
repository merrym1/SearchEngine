import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
LABSE_MODEL = os.getenv("LABSE_MODEL", "sentence-transformers/LaBSE")

class LaBSEEmbedder:
    def __init__(self, model_name: str = None):
        self.model_name = (model_name or LABSE_MODEL).strip()
        print(f"🔹 Memuat model LaBSE: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts, normalize=True, batch_size=64):
        """
        Melakukan embedding terhadap teks menggunakan LaBSE.
        texts: list[str] — bisa berupa query tunggal atau kumpulan teks.
        normalize: apakah hasil vektor dinormalisasi ke panjang 1 (unit vector).
        """
        print(f"🧠 Menghitung embedding untuk {len(texts)} teks ... (normalize={normalize})")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        for i, text in enumerate(texts[:3]): 
            vec = embeddings[i]
            print(f"\n📝 Teks ke-{i+1}: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"📏 Dimensi: {vec.shape[0]}, Norma: {np.linalg.norm(vec):.6f}")
            print(f"🔹 10 komponen pertama: {np.round(vec[:10], 4)}")

        print("✅ Proses embedding selesai.\n")
        return embeddings



