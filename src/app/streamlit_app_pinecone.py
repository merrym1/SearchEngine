import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

import re
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
EMB_PATH = os.getenv("EMB_PATH", "embeddings/embeddings_alkitabbatak.npy")
TOP_K = int(os.getenv("TOP_K", "5"))
USE_PINECONE = os.getenv("USE_PINECONE", "0") == "1"

KITAB_NAMES = [
    "kejadian","keluaran","imamat","bilangan","ulangan",
    "yosua","hakim-hakim","rut",
    "1 samuel","2 samuel","1 raja-raja","2 raja-raja",
    "1 tawarikh","2 tawarikh","ezra","nehemia","ester",
    "ayub","mazmur","amsal","pengkhotbah","kidung agung",
    "yesaya","yeremia","ratapan","yehezkiel","daniel",
    "hosea","joel","amos","obaja","yunus","mikha",
    "nahum","habakuk","zefanya","hagai","zakharia","maleakhi",
    "matius","markus","lukas","yohanes","kisah para rasul",
    "roma","1 korintus","2 korintus","galatia","efesus",
    "filipi","kolose","1 tesalonika","2 tesalonika",
    "1 timotius","2 timotius","titus","filemon",
    "ibrani","yakobus","1 petrus","2 petrus",
    "1 yohanes","2 yohanes","3 yohanes","yudas","wahyu"
]

ROHANI_KEYWORDS = {
    "tuhan","yesus","allah","roh","kudus","iman","kasih",
    "berkat","berkatmu","berkati","berkatnya",
    "doa","berdoa","firman","janji","rahmat","anugerah",
    "juruselamat","penebus","gereja","sorga","surga"
}

def is_rejected_query(q: str) -> str | None:
    """
    Kembalikan pesan error jika query TIDAK valid.
    Jika None → query dianggap OK dan boleh diproses.
    """
    if not q or not q.strip():
        return "Harap isi query terlebih dahulu."

    text = q.strip()
    low = text.lower()
    words = low.split()

    if re.fullmatch(r"[0-9:\s]+", text):
        return "Query tidak boleh hanya berisi angka / nomor ayat. Tulis pernyataan ayatnya."

    if low in KITAB_NAMES:
        return "Query tidak boleh hanya nama kitab. Tambahkan isi ayat/pernyataannya."

    for kitab in KITAB_NAMES:
        k_words = kitab.split()
        if words[:len(k_words)] == k_words and len(words) == len(k_words) + 1:
            if re.fullmatch(r"[0-9:]+", words[-1]):
                return "Query belum cukup. Tulis isi ayat/pernyataannya."
            return "Query belum cukup jelas. Tulis isi ayat/pernyataannya."
        
    if len(words) < 3:
        return "Query minimal 3 kata dalam bentuk pernyataan ayat."

    if len(words) == 3:
        if not any(w in ROHANI_KEYWORDS for w in words):
            return "Query 3 kata harus berkaitan dengan ayat/rohani. Tulis pernyataan ayat yang lebih jelas."

    return None


try:
    from src.vecdb.pinecone_client import pinecone_ready, pinecone_query
except Exception:
    pinecone_ready = lambda: False
    def pinecone_query(*args, **kwargs):
        raise RuntimeError("Pinecone client not available")


st.set_page_config(page_title="Batak Toba Bible Semantic Search", page_icon="🔎", layout="centered")
st.title("🔎 Batak Toba Bible Semantic Search")


with st.sidebar:
    st.header("👤 Informasi Pengembang")
    st.markdown("**Nama:** Merry Uli Artha Manurung  \n**Jurusan:** Teknik Informatika  \n**Skripsi:** Search Engine")


@st.cache_resource(show_spinner=False)
def get_engine(csv_path: str, emb_path: str):
    from src.search.search_engine import SemanticSearchEngine
    return SemanticSearchEngine(csv_path=csv_path, emb_path=emb_path)

if "out" not in st.session_state:
    st.session_state.out = None

def get_normalized_corpus(embs: np.ndarray):
    """Normalisasi embedding corpus dan cache di session."""
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
    """Hitung top-k cosine similarity secara lokal."""
    A = q / (np.linalg.norm(q) + 1e-12)
    sims = Bn @ A
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    idx_top = np.argsort(-sims)[:k]
    return best_score, idx_top, sims


def ann_topk_with_pinecone(q_emb: np.ndarray, k: int):
    """Wrapper Pinecone query → DataFrame hasil."""
    matches = pinecone_query(q_emb, top_k=k)
    df = pd.DataFrame(matches)
    if "score" in df.columns:
        df = df.rename(columns={"score": "score"})
    return df


with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Masukkan pernyataan (Bahasa Indonesia):",
        placeholder="Contoh: Kasih Tuhan yang besar bagi umat-Nya",
        value=""
    )
    k = st.slider("Top-K hasil:", min_value=3, max_value=30, value=TOP_K, step=1)
    threshold = 0.7
    submitted = st.form_submit_button("Cari")


if submitted:
    try:

        msg = is_rejected_query(query)
        if msg:
            st.warning(msg)
            st.stop()

        if not os.path.exists(CSV_PATH):
            st.error(f"CSV tidak ditemukan: `{CSV_PATH}`"); st.stop()
        if not os.path.exists(EMB_PATH):
            st.error(f"Embedding NPY tidak ditemukan: `{EMB_PATH}`"); st.stop()


        t0 = time.time()
        with st.spinner("⚙️ Memuat engine (sekali di awal)..."):
            engine = get_engine(CSV_PATH, EMB_PATH)
        t1 = time.time()


        with st.spinner("🌐 Menerjemahkan (NLLB)..."):
            out_tuple = engine.query(query, top_k=1)
            if not (isinstance(out_tuple, (tuple, list)) and len(out_tuple) >= 1):
                raise RuntimeError("engine.query() tidak mengembalikan hasil terjemahan.")
            translated = out_tuple[0]

        from src.models.embed_labse import LaBSEEmbedder
        labse = LaBSEEmbedder()

        variants = [
            ("nllb_translated", translated),
            ("indonesian_raw", query),
            ("nllb_translated_lower", translated.lower()),
            ("indonesian_raw_lower", query.lower()),
        ]

        Bn, stats = get_normalized_corpus(engine.store.embs)
        best_global, best_pack = -1.0, None

        t2 = time.time()
        with st.spinner("🔍 Menghitung embedding & mencari hasil terbaik..."):
            for vname, vtext in variants:
                q_emb = labse.encode([vtext], normalize=True)[0]  

                if USE_PINECONE and pinecone_ready():

                    df = ann_topk_with_pinecone(q_emb, k)
                    if not df.empty:
                        top_score = float(df["score"].max())
                        if top_score > best_global:
                            best_global = top_score
                            best_pack = (vname, q_emb, top_score, df)
                else:

                    score, idx_top, sims = cosine_topk(Bn, q_emb, k)
                    if score > best_global:
                        best_global = score
                        results = engine.store.csv.iloc[idx_top].copy()
                        results["score"] = sims[idx_top]
                        results = results.reset_index(drop=True)
                        best_pack = (vname, q_emb, score, results)

                if best_global >= threshold:
                    break

        if not best_pack:
            st.error("❌ Tidak ada hasil ditemukan. Periksa koneksi Pinecone atau dataset lokal.")
            st.stop()

        vname, best_qemb, best_score, results_df = best_pack
        t3 = time.time()

        st.session_state.out = {
            "translated": translated,
            "best_variant": vname,
            "q_vec": [float(x) for x in np.asarray(best_qemb).ravel().tolist()],
            "results_records": results_df.head(k).to_dict(orient="records"),
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
            "backend": "pinecone" if (USE_PINECONE and pinecone_ready()) else "local",
        }

    except Exception as e:
        logger.exception("❌ Gagal menjalankan pipeline")
        st.error("Terjadi error saat menjalankan pencarian:")
        st.exception(e)
        st.stop()


out = st.session_state.out
if out:
    t = out["timing"]
    st.caption(f"⏱️ Load: {t['load_ms']} ms | Embed: {t['embed_ms']} ms | Search: {t['search_ms']} ms | Total: {t['total_ms']} ms")

    st.subheader("Terjemahan Bahasa Batak Toba")
    st.code(out["translated"])

    st.subheader("Hasil Pencarian")
    df = pd.DataFrame(out["results_records"])
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        for i, row in df.iterrows():
            kitab = row.get("kitab", "")
            pasal = row.get("pasal", "")
            ayat = row.get("ayat", "")
            teks = row.get("teks", "")
            score = float(row.get("score", 0))
            st.markdown(f"**{i+1}. {kitab} {pasal}:{ayat}**")
            st.write(teks)
            st.caption(f"Skor kesamaan: `{score:.4f}`")
            st.markdown("---")
