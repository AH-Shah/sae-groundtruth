"""Checks the BatchTopK inference threshold on the 1M-token language-model data (Methods).
"""
import os, sys, time, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("SAEGT_DATA", ROOT / "data")); RUNS = Path(os.environ.get("SAEGT_OUT", ROOT / "results/runs")); RUNS.mkdir(parents=True, exist_ok=True)
from saegt import sae as dense
z = np.load(DATA / "inputs/LLM_pythia70m_L3_1M.npz"); X = z["latent"].astype(np.float32)
X -= X.mean(0); X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9); X *= 1 / np.sqrt((X ** 2).mean())
rng = np.random.default_rng(0); ev = np.sort(rng.choice(len(X), 100000, replace=False)); fm = np.ones(len(X), bool); fm[ev] = False; XF, EV = X[fm], X[ev]; del X
M, k = 2048, 128; t0 = time.time(); m, h = dense.fit(XF, dict(M=M, k=k, steps=3000, matryoshka=False, seed=0, device="mps")); print(f"fit {time.time() - t0:.0f}s", flush=True)
sub = XF[np.random.default_rng(1).choice(len(XF), 100000, replace=False)]
with torch.no_grad(): A = m.pre(torch.as_tensor(sub)).numpy()
thr_exact = float(np.partition(A.ravel(), -k * len(A))[-k * len(A)])
L0 = lambda Z: float((Z != 0).sum(1).mean())
print(f"stored thr {float(m.thr):.5f} | exact thr (CPU, 100k fit rows) {thr_exact:.5f} | L0 fit rows {L0(dense.encode(m, sub)):.1f} | L0 held-out {L0(dense.encode(m, EV)):.1f} | "
      f"L0 held-out at exact thr {float((m.pre(torch.as_tensor(EV)).detach().numpy() > thr_exact).sum(1).mean()):.1f} | n*k {k * len(XF):,} of {len(XF) * M:,} entries", flush=True)
