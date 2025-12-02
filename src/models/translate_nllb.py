import os, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = os.getenv("NLLB_MODEL", "kepinsam/ind-to-bbc-nmt-v5")

class NLLBTranslator:
    def __init__(self, model_name=None, device=None):
        self.model_name = (model_name or MODEL_NAME).strip()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()


    @torch.no_grad()
    def translate(self, text: str, num_beams: int = 6, max_new_tokens: int = 96) -> str:
        """Deterministik — tanpa sampling, hasil selalu sama untuk input yang sama."""
        text = (text or "").strip()
        if not text:
            return ""
        inp = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        out_ids = self.model.generate(
            **inp,
            num_beams=num_beams,          
            do_sample=False,               
            max_new_tokens=max_new_tokens,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    @torch.no_grad()
    def translate_n(self, text: str, n: int = 6) -> list[str]:
        """Hanya untuk eksplorasi — gunakan jika ingin beberapa versi terjemahan."""
        text = (text or "").strip()
        if not text:
            return []
        inp = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        out_ids = self.model.generate(
            **inp,
            num_beams=1,
            do_sample=True,         
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            max_new_tokens=96,
            num_return_sequences=n,
            no_repeat_ngram_size=3,
        )
        outs = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        seen, uniq = set(), []
        for s in outs:
            s = (s or "").strip()
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq


