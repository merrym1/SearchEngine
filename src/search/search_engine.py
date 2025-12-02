from src.search.vector_store import VectorStore
from src.models.translate_nllb import NLLBTranslator
from src.models.embed_labse import LaBSEEmbedder

class SemanticSearchEngine:
    def __init__(self, csv_path: str, emb_path: str, nllb_model: str = None, labse_model: str = None):
        self.store = VectorStore(csv_path, emb_path)
        self.translator = NLLBTranslator(model_name=nllb_model)
        self.embedder = LaBSEEmbedder(model_name=labse_model)

    def query(self, text_id: str, top_k: int = 10):
        translated = self.translator.translate(text_id)
        q_emb = self.embedder.encode([translated], normalize=True)[0]
        results = self.store.search(q_emb, top_k=top_k) 
        return translated, q_emb, results
