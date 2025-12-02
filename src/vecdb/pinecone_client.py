# src/vecdb/pinecone_client.py
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

_api = os.getenv("PINECONE_API_KEY", "pcsk_3fY3du_CDsFipaY8jALNt9Yx8TFrf8aPo3NnBPvkdtmzt4gCLq5ngdBUxHspgEZGUa39bR")
_name = os.getenv("PINECONE_INDEX", "bible-batak-labse")
_pc = Pinecone(api_key=_api) if _api else None
_index = _pc.Index(_name) if (_pc and _name) else None

def pinecone_ready() -> bool:
    return _index is not None

def pinecone_query(q_vec, top_k: int = 10, kitab: Optional[str] = None) -> List[Dict[str, Any]]:
    assert pinecone_ready(), "pcsk_3fY3du_CDsFipaY8jALNt9Yx8TFrf8aPo3NnBPvkdtmzt4gCLq5ngdBUxHspgEZGUa39bR"
    flt = {"kitab": {"$eq": kitab}} if kitab else None
    res = _index.query(
        vector=q_vec.tolist() if hasattr(q_vec, "tolist") else list(q_vec),
        top_k=top_k,
        filter=flt,
        include_metadata=True
    )
    out = []
    for m in res["matches"]:
        md = m.get("metadata", {}) or {}
        out.append({
            "kitab": md.get("kitab", ""),
            "pasal": int(float(md.get("pasal", 0))) if md.get("pasal", "") != "" else "",
            "ayat":  int(float(md.get("ayat", 0)))  if md.get("ayat", "")  != "" else "",
            "teks":  md.get("teks", ""),
            "score": float(m["score"]),
        })
    return out
