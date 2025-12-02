import numpy as np
import pandas as pd

class VectorStore:
    def __init__(self, csv_path: str, emb_path: str, assume_normalized: bool = True):
        """
        assume_normalized:
          - True  -> embeddings di NPY sudah dinormalisasi (umum pada LaBSE).
          - False -> akan dinormalisasi di sini saat load.
        """
        self.csv = pd.read_csv(csv_path)
        self.embs = np.load(emb_path)

        assert len(self.csv) == self.embs.shape[0], \
            f"CSV rows ({len(self.csv)}) != Embeddings rows ({self.embs.shape[0]})"

        if not assume_normalized:
            norms = np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-12
            self.embs = self.embs / norms

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def cosine_all(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Hitung cosine similarity ke semua dokumen.
        Asumsi: self.embs sudah normalized; query_emb akan dinormalisasi di sini.
        """
        A = self._normalize(query_emb.ravel())
        B = self.embs  
        return B @ A  

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 10,
        threshold: float | None = None,
        return_meta: bool = False
    ):
        """
        Kembalikan top_k hasil + (opsional) meta:
          meta = {
              "best_idx": int,
              "best_score": float,
              "best_row": dict,   # satu baris CSV untuk skor terbaik
              "threshold": float|None,
              "pass_threshold": bool
          }
        """
        sims = self.cosine_all(query_emb)            
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

    
        idx_top = np.argsort(-sims)[:top_k]
        results = self.csv.iloc[idx_top].copy()
        results["score"] = sims[idx_top]
        results = results.reset_index(drop=True)

        if not return_meta:
            return results

        meta = {
            "best_idx": best_idx,
            "best_score": best_score,
            "best_row": self.csv.iloc[best_idx].to_dict(),
            "threshold": threshold,
            "pass_threshold": (threshold is not None and best_score >= threshold),
        }
        return results, meta

