import torch
from collections import defaultdict

def _norm_edges(edges):
    """Normalize edges to the format (src, edge_type, dst)."""
    norm = []
    for e in edges:
        if isinstance(e, (list, tuple)):
            if len(e) == 3:
                s, et, d = e
                norm.append((int(s), str(et), int(d)))
            elif len(e) == 2:
                s, d = e
                norm.append((int(s), "E", int(d)))
    return norm

def _build_incidence(n_nodes, hyperedges_sets):
    """hyperedges_sets: list[ set(node_ids) ]"""
    E = len(hyperedges_sets)
    H = torch.zeros((n_nodes, E), dtype=torch.float32)
    De = torch.zeros(E, dtype=torch.float32)
    Dv = torch.zeros(n_nodes, dtype=torch.float32)
    for ei, nodes in enumerate(hyperedges_sets):
        if not nodes:
            continue
        De[ei] = float(len(nodes))
        for v in nodes:
            H[v, ei] = 1.0
            Dv[v] += 1.0
    # Avoid division by zero.
    De[De == 0] = 1.0
    Dv[Dv == 0] = 1.0
    return H, De, Dv

def make_hyperedges(
    n_nodes,
    edges,
    k_call_ctx=3,
    d_struct=3,
    use_pairwise=True,
    use_struct=True,
    use_coperm=True,
    use_callctx=True,
):
    """
    Three optional semantic hyperedge families plus pairwise hyperedges:
    - structural: aggregate a first-order neighborhood, taking at most d_struct neighbors.
    - co-permission: group nodes with the same in/out-degree signature as a proxy when explicit
      privilege annotations are unavailable.
    - call-context: aggregate up to k_call_ctx outgoing neighbors of high-outdegree nodes.
    - pairwise: keep original pairwise edges as size-2 hyperedges; this is recommended.
    """
    edges = _norm_edges(edges)

    outN = defaultdict(set)
    inN = defaultdict(set)
    for s, et, d in edges:
        outN[s].add(d)
        inN[d].add(s)

    hyperedges = []

    # A) structural hyperedges
    if use_struct:
        for v in range(n_nodes):
            nbrs = outN[v] | inN[v]
            if not nbrs:
                continue
            S = set([v]) | set(list(nbrs)[:d_struct])
            if len(S) >= 2:
                hyperedges.append(S)

    # B) co-permission hyperedges (approx by degree signature)
    if use_coperm:
        deg_sig = defaultdict(list)
        for v in range(n_nodes):
            deg_sig[(len(inN[v]), len(outN[v]))].append(v)
        for sig, nodes in deg_sig.items():
            if len(nodes) >= 3:
                hyperedges.append(set(nodes))

    # C) call-context hyperedges
    if use_callctx:
        for v in range(n_nodes):
            if len(outN[v]) >= 2:
                ctx = set([v]) | set(list(outN[v])[:k_call_ctx])
                if len(ctx) >= 2:
                    hyperedges.append(ctx)

    # D) keep pairwise hyperedges (do NOT remove)
    if use_pairwise:
        for s, et, d in edges:
            if s != d:
                hyperedges.append(set([s, d]))

    # de-duplicate
    uniq = []
    seen = set()
    for S in hyperedges:
        key = tuple(sorted(S))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(S)

    return _build_incidence(n_nodes, uniq)

