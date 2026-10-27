import numpy as np


def evaluate(predictions, truth, users, k=100):
    expected = {}
    for row in truth.select("uid", "item_id").iter_rows():
        expected.setdefault(row[0], set()).add(row[1])
    vals, full, active = [], [], []
    for u in users:
        t = expected.get(u, set())
        r = len(t & set(predictions.get(u, [])[:k])) / min(len(t), k) if t else 0.
        vals.append(r)
        if t: active.append(r)
        if len(t) >= 100: full.append(r)
    return {"recall_all": float(np.mean(vals)) if vals else 0.,
            "recall_active": float(np.mean(active)) if active else None,
            "recall_full100": float(np.mean(full)) if full else None,
            "n_users": len(vals), "n_active": len(active), "n_full100": len(full),
            "n_zero_future": len(vals)-len(active)}


def candidate_report(candidates, truth, users):
    pred = {u: g["item_id"].to_list() for (u,), g in candidates.group_by("uid")}
    # Actual relevant fraction and the attainable final Recall@100 are different for >100 labels.
    relevant = {}; out = []; ceiling = []
    for u, item in truth.select("uid", "item_id").iter_rows():
        relevant.setdefault(u, set()).add(item)
    for u in users:
        t = relevant.get(u, set())
        if not t: continue
        hits = len(t & set(pred.get(u, [])))
        out.append(hits / len(t))
        ceiling.append(min(hits, 100) / min(len(t), 100))
    return {"candidate_relevant_fraction_active": float(np.mean(out)) if out else None,
            "oracle_recall100_active": float(np.mean(ceiling)) if ceiling else None,
            "candidate_pairs": candidates.height}


def paired_bootstrap(a, b, truth, users, repeats=1000, seed=42):
    t = {}
    for u, i in truth.select("uid", "item_id").iter_rows(): t.setdefault(u, set()).add(i)
    d = np.array([(len(set(a.get(u, [])[:100]) & t[u])-len(set(b.get(u, [])[:100]) & t[u])) / min(len(t[u]),100)
                  for u in users if t.get(u)])
    if not len(d): return None
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(repeats)])
    return {"delta_active": float(d.mean()), "ci95": np.quantile(means, [.025,.975]).tolist(), "n":len(d)}
