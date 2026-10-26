"""[Ported from antig/N2/tools/84_llm_embed.py by tools/97_port_scripts.py: output path (paper: LLM_N=1000000 LLM_SPLIT=train-00000-of-00002 LLM_OUT=LLM_pythia70m_L3_1M.npz; preliminary: defaults)]
row 333 LLM leg: Pythia-70m residual stream (layer 3, d 512) on WikiText-103 validation; rule-based token classes as 'concepts'.
-> N2/trials/dsae/LLM_pythia70m_L3.npz (latent, cls, tok). CPU."""
import re, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
N = int(os.environ.get("LLM_N", "50000")); SPLIT = os.environ.get("LLM_SPLIT", "validation-00000-of-00001")  # default-off (row 338): 1M train tokens
ROOT = Path(__file__).resolve().parents[1]; OUT = str(Path(os.environ.get("SAEGT_DATA", ROOT / "data")) / "inputs" / os.environ.get("LLM_OUT", "LLM_pythia70m_L3.npz")); CTX, SKIP, LAYER = 128, 8, 3
FUNC = set("the of and to a in is was for on that with as by at from it an his her their its be are were or which this but not had have has he she they we".split())


def cls(t):
    sp = t.startswith("Ġ"); w = t.lstrip("Ġ")
    if not w or not re.search(r"\w", w): return "PUNCT" if w else "OTH"
    if w.isdigit(): return "NUM"
    if not w.isalpha(): return "OTH"
    if not sp: return "CONT"
    if w.lower() in FUNC: return "FUNC"
    if w[0].isupper(): return "CAP"
    for s, c in (("ing", "ING"), ("ed", "ED"), ("ly", "LY")):
        if w.endswith(s) and len(w) > len(s) + 2: return c
    return "OTH"


f = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{SPLIT}.parquet", repo_type="dataset")
text = " ".join(x.strip() for x in pd.read_parquet(f).text if x.strip() and not x.strip().startswith("="))
tk = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m"); m = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m").eval()
ids = tk(text[:max(2_000_000, N * 6)])["input_ids"]; per = CTX - SKIP; nw = -(-N // per); H, T = [], []
with torch.no_grad():
    for i in range(nw):
        w = ids[i * CTX:(i + 1) * CTX]
        if len(w) < CTX: break
        h = m(torch.tensor([w]), output_hidden_states=True).hidden_states[LAYER][0, SKIP:]; H.append(h.numpy().astype(np.float16)); T += tk.convert_ids_to_tokens(w[SKIP:])
Z = np.vstack(H)[:N].astype(np.float32); T = np.array(T[:N]); C = np.array([cls(t) for t in T])
np.savez_compressed(OUT, latent=Z, cls=C, tok=T); print(Z.shape, dict(zip(*np.unique(C, return_counts=True))))
