import json, hashlib
from pathlib import Path
import numpy as np
import query_consensus_v10 as c
R = Path(__file__).resolve().parent
O = R / 'posterior_v16'
O.mkdir(exist_ok=True)

def residual(P):
    return float(max(np.max(abs(P.sum(0) - 1)), np.max(abs(P.sum(1) - 1))))

def rounded(P):
    X = np.asarray(P, dtype=np.float64).copy()
    assert np.all(np.isfinite(X)) and X.min() >= 0
    X *= np.minimum(1, 1 / np.maximum(X.sum(1), 1e-300))[:, None]
    X *= np.minimum(1, 1 / np.maximum(X.sum(0), 1e-300))[None, :]
    a = np.maximum(0, 1 - X.sum(1))
    b = np.maximum(0, 1 - X.sum(0))
    mass = a.sum()
    if mass > 1e-14:
        X += np.outer(a, b) / mass
    assert X.min() >= 0 and residual(X) < 1e-10
    return X

def main():
    rng = np.random.default_rng(1601)
    check = []
    for case in range(100):
        n = int(rng.integers(2, 100))
        P = np.zeros((n, n)) if case == 0 else rng.random((n, n)) * 10 ** rng.uniform(-20, 20)
        B = rounded(P)
        check.append(dict(case=case, n=n, residual=residual(B)))
    (O / 'checks.json').write_text(json.dumps(check, indent=2))
    rows = []
    for folder in ['desalign_v9', 'desalign_v11', 'desalign_v11_noimage_fbdb']:
        for file in sorted((R / folder).glob('*_scores.npz')):
            z = np.load(file)
            U = z['U'].astype(bool)
            G = c.answers(z['gold'], U)
            P, before = c.sinkhorn(z['depth2'], 0.05)
            B = rounded(P)
            l1 = float(np.abs(B - P).sum())
            for mode, X in [('fixed500', P), ('marginal_repaired', B)]:
                q = U.astype(float) @ X
                clipped = np.clip(q, 1e-12, 1 - 1e-12)
                p = c.assignment(U.T.astype(float) @ q)
                rows.append(dict(folder=folder, file=file.name, mode=mode, marginal_residual=residual(X), repair_matrix_l1=l1, query_brier=float(np.mean((q - G) ** 2)), query_logloss=float(-np.mean(G * np.log(clipped) + ~G * np.log(1 - clipped))), query_loss=c.summarize(c.answers(p, U), U, G)['query_loss']))
    (O / 'results.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(dict(matrix_conditions=len(rows), checks=len(check), max_checked_residual=max((r['residual'] for r in check)), max_input_residual=max((r['marginal_residual'] for r in rows if r['mode'] == 'fixed500')), max_repaired_residual=max((r['marginal_residual'] for r in rows if r['mode'] == 'marginal_repaired'))), indent=2))
if __name__ == '__main__':
    main()
