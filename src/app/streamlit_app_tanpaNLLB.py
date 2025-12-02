import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

import os, sys, time, io, logging
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


LOG_LEVEL = logging.INFO
logger = logging.getLogger("batak-search")
if not logger.handlers:
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(sh)
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler("logs/app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)


load_dotenv()
CSV_PATH = os.getenv("CSV_PATH", "data/processed/hasil_newfinalparsing_progress3_allhal.csv")
EMB_PATH = os.getenv("EMB_PATH", os.path.join("embeddings", "embeddings_alkitabbatak.npy"))
TOP_K    = int(os.getenv("TOP_K", "5"))


st.set_page_config(page_title="Batak Toba Bible Semantic Search", page_icon="🔎", layout="centered")
st.title("🔎 Batak Toba Bible Semantic Search")
st.caption("Query → Embed (LaBSE) → Retrieve (Cosine)")

with st.sidebar:
    st.header("👤 Informasi Pengembang")
    st.markdown("**Nama:** Merry Uli Artha Manurung  \n**Jurusan:** Teknik Informatika   \n**Skripsi:** Search Engine ")


@st.cache_resource(show_spinner=False)
def get_engine(csv_path: str, emb_path: str):
    from src.search.search_engine import SemanticSearchEngine
    return SemanticSearchEngine(csv_path=csv_path, emb_path=emb_path)

if "out" not in st.session_state:
    st.session_state.out = None


def get_normalized_corpus(embs: np.ndarray):
    if "corpus_norm" in st.session_state and st.session_state.get("corpus_shape") == embs.shape:
        return st.session_state["corpus_norm"], st.session_state["corpus_stats"]
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    normed = embs / norms
    stats = {
        "mean_norm_before": float(np.mean(np.linalg.norm(embs, axis=1))),
        "mean_norm_after": float(np.mean(np.linalg.norm(normed, axis=1))),
    }
    st.session_state["corpus_norm"] = normed
    st.session_state["corpus_shape"] = embs.shape
    st.session_state["corpus_stats"] = stats
    logger.info("Corpus normalized: mean_norm_before=%.4f → after=%.4f",
                stats["mean_norm_before"], stats["mean_norm_after"])
    return normed, stats


def cosine_topk(Bn: np.ndarray, q: np.ndarray, k: int):
    A = q / (np.linalg.norm(q) + 1e-12)
    sims = Bn @ A  # cosine ∈ [-1, 1]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    idx_top = np.argsort(-sims)[:k]
    return best_score, idx_top, sims


with st.form("search_form", clear_on_submit=False):
    query = st.text_input("Masukkan pernyataan (Bahasa Indonesia):", value="Minimal 3 kata")
    k = st.slider("Top-K hasil:", min_value=3, max_value=30, value=TOP_K, step=1)
    threshold = 0.6
    submitted = st.form_submit_button("Cari")

if submitted:
    try:
        if not query.strip():
            st.warning("Harap isi query terlebih dahulu.")
            st.stop()

        if not os.path.exists(CSV_PATH):
            st.error(f"CSV tidak ditemukan: `{CSV_PATH}`"); st.stop()
        if not os.path.exists(EMB_PATH):
            st.error(f"Embedding NPY tidak ditemukan: `{EMB_PATH}`"); st.stop()


        t0 = time.time()
        with st.spinner("Memuat korpus…"):
            engine = get_engine(CSV_PATH, EMB_PATH)
        t1 = time.time()

        from src.models.embed_labse import LaBSEEmbedder
        labse = LaBSEEmbedder()


        variants = [
            ("indonesian_raw", query),
            ("indonesian_raw_lower", query.lower()),
        ]


        Bn, stats = get_normalized_corpus(engine.store.embs)

        best_global = -1.0
        best_pack = None  

        t2 = time.time()
        with st.spinner("Menghitung embedding & mencari yang paling mirip…"):
            for vname, vtext in variants:
                q_emb = labse.encode([vtext], normalize=True)[0] 
                score, idx_top, sims = cosine_topk(Bn, q_emb, k)
                logger.info("Variant=%s | best=%.6f", vname, score)
                if score > best_global:
                    best_global = score
                    best_pack = (vname, q_emb, score, idx_top, sims)
                if best_global >= threshold:
                    break

 
        assert best_pack is not None
        vname, best_qemb, best_score, idx_top, sims = best_pack
        results = engine.store.csv.iloc[idx_top].copy()
        results["score"] = sims[idx_top]
        results = results.reset_index(drop=True)

        t3 = time.time()
        logger.info("DONE | variant=%s | best=%.6f | load=%d ms | search=%d ms | total=%d ms",
                    vname, best_score,
                    int((t1 - t0)*1000),
                    int((t3 - t2)*1000),
                    int((t3 - t0)*1000))


        st.session_state.out = {
            "best_variant": vname,
            "q_vec": [float(x) for x in np.asarray(best_qemb).ravel().tolist()],
            "results_records": results.head(k).to_dict(orient="records"),
            "timing": {
                "load_ms": int((t1 - t0) * 1000),
                "embed_ms": int((t2 - t1) * 1000),
                "search_ms": int((t3 - t2) * 1000),
                "total_ms": int((t3 - t0) * 1000),
            },
            "best_score": float(best_score),
            "best_threshold": float(threshold),
            "best_pass": (best_score >= float(threshold)),
            "corpus_stats": stats,
            "raw_query": query,
        }

    except Exception as e:
        logger.exception("Gagal menjalankan pipeline")
        st.error("❌ Terjadi error saat memuat/menjalankan pencarian:")
        st.exception(e)
        st.stop()


out = st.session_state.out
logger.info("Render — out is %s", "SET" if out else "EMPTY")

if out:
    t = out["timing"]
    st.caption(
        f"⏱️ Load: {t['load_ms']} ms | Embed: {t['embed_ms']} ms | "
        f"Search: {t['search_ms']} ms | Total: {t['total_ms']} ms"
    )

    st.subheader("Hasil Pencarian")
    df = pd.DataFrame(out["results_records"])
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        cols = set(df.columns)
        c_book  = "kitab" if "kitab" in cols else ("book" if "book" in cols else None)
        c_ch    = "pasal" if "pasal" in cols else ("chapter" if "chapter" in cols else None)
        c_vs    = "ayat"  if "ayat"  in cols else ("verse"   if "verse"   in cols else None)
        c_text  = "teks"  if "teks"  in cols else ("text"    if "text"    in cols else None)
        c_score = "score" if "score" in cols else ("similarity" if "similarity" in cols else None)

        if not c_text or not c_score:
            st.warning(f"Kolom tersedia: {list(cols)} — butuh minimal teks/text dan score/similarity.")
        else:
            for i, row in df.iterrows():
                book = row[c_book] if c_book else ""
                ch   = row[c_ch]   if c_ch   else ""
                vs   = row[c_vs]   if c_vs   else ""
                txt  = row[c_text]
                scr  = float(row[c_score])
                title = f"**{i+1}. {book} {ch}:{vs}**" if (book and ch and vs) else f"**{i+1}.**"
                st.markdown(title)
                st.write(txt)
                st.caption(f"Skor kesamaan: `{scr:.4f}`")
                st.markdown("---")
