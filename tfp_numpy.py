import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def norm(a):
    return a / np.maximum(np.sqrt(np.sum(a * a, axis=1, keepdims=True)), 1e-12)

def rownorm(A):
    s = np.asarray(A.sum(1)).ravel()
    return A.multiply(1 / np.maximum(s[:, None], 1)).tocsr()

def graph(triples, n):
    T = np.unique(np.r_[triples[:, [0, 2, 1]] * np.array([1, 1, 2]), triples[:, [2, 0, 1]] * np.array([1, 1, 2]) + np.array([0, 0, 1])], axis=0)
    nr = int(T[:, 2].max()) + 1
    edges, edgeid = np.unique(T[:, :2], axis=0, return_inverse=True)
    weights = {(i, i): 0 for i in range(n)}
    last = (-1, -1)
    for h, t, r in T:
        weights[h, h] += 1
        weights[t, t] += 1
        if (h, t) != last:
            last = (h, t)
            weights[h, t] = 0
        weights[h, t] += 1
    ij = np.array(list(weights))
    A = coo_matrix((np.array(list(weights.values()), np.float32), ij.T), shape=(n, n)).tocsr()
    head_pairs = np.unique(T[:, [2, 0]], axis=0)
    H = coo_matrix((np.ones(len(head_pairs), np.float32), head_pairs.T), shape=(nr, n)).tocsr()
    E = coo_matrix((np.ones(len(T), np.float32), (edgeid, T[:, 2])), shape=(len(edges), nr)).tocsr()
    return (rownorm(A).astype(np.float32), rownorm(H).astype(np.float32), E, edges)

def projection(rng, dim, out):
    return norm(rng.normal(0, (1 / out) ** 0.5, (dim, out)).astype(np.float32))

def propagate(initial, train, triples, selected, rel_dim=512, mini_dim=16, seed=24001, dense_reference=False, project_first=False):
    rng = np.random.default_rng(seed)
    n = len(initial)
    A, H, E, edges = graph(triples, n)
    new = A @ initial
    new[train.ravel()] = initial[train.ravel()]
    F = norm(np.c_[initial, norm(new)])
    rel = norm(H @ initial)
    rel = norm(np.c_[np.zeros_like(rel), rel])
    relproj = rel @ projection(rng, rel.shape[1], rel_dim)
    result = np.empty((len(selected), rel_dim * mini_dim), np.float32)
    for h in range(rel_dim):
        weight = E @ relproj[:, h]
        B = coo_matrix((weight, edges.T), shape=(n, n)).tocsr()
        P = projection(rng, F.shape[1], mini_dim)
        result[:, h * mini_dim:(h + 1) * mini_dim] = (B.toarray() @ (F @ P))[selected] if dense_reference else B[selected] @ (F @ P) if project_first else B[selected] @ F @ P
    return np.c_[F[selected], norm(result)].astype(np.float32)

def check():
    rng = np.random.default_rng(24000)
    errors = []
    for n in [9, 15, 23]:
        T = np.c_[rng.integers(n, size=40), rng.integers(4, size=40), rng.integers(n, size=40)]
        T = np.r_[T, T[:3]]
        x = norm(rng.normal(size=(n, 7)).astype(np.float32))
        train = np.array([[0, n - 1], [1, n - 2]])
        selected = np.array([0, 2, 3, n - 1])
        a = propagate(x, train, T, selected, rel_dim=9, mini_dim=3)
        b = propagate(x, train, T, selected, rel_dim=9, mini_dim=3, dense_reference=True)
        err = float(abs(a - b).max())
        assert err < 1e-05
        errors.append(err)
    return dict(sizes=[9, 15, 23], max_selected_row_algebra_error=max(errors), comparison='Dense direct versus selected-row associative execution; this does not compare against a TensorFlow runtime')
