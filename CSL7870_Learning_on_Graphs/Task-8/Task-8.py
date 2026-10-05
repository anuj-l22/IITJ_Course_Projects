import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import collections
import numpy as np
import random
import networkx as nx
from itertools import combinations

def weisfeiler_lehman_hash(graph):
    colors = {node: data.get('label', 0) for node, data in graph.nodes(data=True)}

    for _ in range(len(graph.nodes())): # Iterate enough times for convergence
        new_colors = {}
        for node in graph.nodes():
            neighbor_colors = sorted([colors[nbr] for nbr in graph.neighbors(node)])
            signature = (colors[node], tuple(neighbor_colors))
            new_colors[node] = hash(signature) # Hash the signature to get a new color

        if new_colors == colors:
            break
        colors = new_colors

    canonical_hash = str(sorted(colors.values()))
    return canonical_hash

def find_isomorphic_groups_from_pyg(dataset):
    hashes = collections.defaultdict(list)

    for i, data in enumerate(dataset):
        # g = to_networkx(data, node_attrs=['x'])
        g = to_networkx(data, node_attrs=['x'], to_undirected=True)
        g.remove_edges_from(nx.selfloop_edges(g))
        bc = nx.betweenness_centrality(g, normalized=False)   # raw counts
        bc_int = {u: int(round(bc[u])) for u in g.nodes()}    # make sure integer
        nx.set_node_attributes(g, bc_int, 'label')
        if len(g) > 0:
            h = weisfeiler_lehman_hash(g)
            hashes[h].append(i)
            
    isomorphic_groups = {h: indices for h, indices in hashes.items() if len(indices) > 1}
    return isomorphic_groups

print("1. Loading AIDS dataset using PyTorch Geometric...")
try:
        # use_node_attr=True loads the node labels into the `x` attribute
    dataset = TUDataset(root='/tmp/AIDS', name='AIDS', use_node_attr=True)
    print(f"   Successfully loaded {len(dataset)} graphs.")
except Exception as e:
    print(f"Error: Could not download or load the dataset. {e}")
    print("Please check your internet connection and if the TUDataset repository is accessible.")
    exit()

print("\n2. Running Weisfeiler-Lehman isomorphism test...")
isomorphic_groups = find_isomorphic_groups_from_pyg(dataset)
print("   Test complete.")

print("\n3. Isomorphism Statistics:")
if not isomorphic_groups:
    print("   No isomorphic graphs were found in the dataset.")
else:
    num_isomorphic_groups = len(isomorphic_groups)
    total_graphs_in_groups = sum(len(indices) for indices in isomorphic_groups.values())
    print(f"   - Total Isomorphic Groups Found: {num_isomorphic_groups}")
    print(f"   - Total Graphs in Isomorphic Groups: {total_graphs_in_groups}")

    print("\n   Isomorphic Group Details (Graph Indices):")
        # Sort groups by the first graph index for consistent output
    sorted_groups = sorted(isomorphic_groups.values(), key=lambda x: x[0])
    for i, group in enumerate(sorted_groups):
        print(f"     - Group {i+1}: {group}")


def random_single_edge_perturbation(G: nx.Graph):
    """
    Randomly EITHER add one non-existent edge OR remove one existing edge.
    Exactly one change is made.
    Returns (H, op, edge).
    """
    H = G.copy()
    nodes = list(H.nodes())
    edges = {tuple(sorted(e)) for e in H.edges()}
    nonedges = [e for e in combinations(nodes, 2) if e not in edges]

    can_add = len(nonedges) > 0
    can_remove = len(edges) > 0

    if can_add and can_remove:
        op = random.choice(["add", "remove"])
    elif can_add:
        op = "add"
    elif can_remove:
        op = "remove"

    if op == "add":
        e = random.choice(nonedges)
        H.add_edge(*e)
    else:
        e = random.choice(list(edges))
        H.remove_edge(*e)

    # sanity check: must differ by exactly one edge
    E1 = {tuple(sorted(ed)) for ed in G.edges()}
    E2 = {tuple(sorted(ed)) for ed in H.edges()}
    diff = E1 ^ E2
    if len(diff) != 1:
        raise ValueError("Perturbation failed: more than one change detected.")

    return H, op, e

# ---------- helpers (same labeling & WL as in your pipeline) ----------

def to_nx_with_bc_int(data):
    """PyG Data -> NetworkX (undirected, no self-loops) with 'label' = int(round(bc))."""
    g = to_networkx(data, node_attrs=['x'], to_undirected=True)
    g.remove_edges_from(nx.selfloop_edges(g))
    bc = nx.betweenness_centrality(g, normalized=False)
    bc_int = {u: int(round(bc[u])) for u in g.nodes()}
    nx.set_node_attributes(g, bc_int, 'label')
    return g

def relabel_with_bc_int(G):
    """Recompute betweenness-int labels on an existing NetworkX graph copy."""
    H = G.copy()
    bc = nx.betweenness_centrality(H, normalized=False)
    labels = {v: int(round(bc[v])) for v in H.nodes()}
    nx.set_node_attributes(H, labels, 'label')
    return H

def extract_largest_group(isomorphic_groups):
    """Return (group_hash, [indices]) for the largest WL group."""
    group_hash = max(isomorphic_groups, key=lambda h: len(isomorphic_groups[h]))
    return group_hash, isomorphic_groups[group_hash]


# ---------- Experiment A flow ----------

# 1) pick largest WL-isomorphic group
if not isomorphic_groups:
    print("No WL-isomorphic groups found; nothing to perturb.")
else:
    group_hash, group_indices = extract_largest_group(isomorphic_groups)
    print(f"\nLargest WL group hash: {group_hash}")
    print(f"Group size: {len(group_indices)}")
    print(f"Graph indices: {group_indices}")

    # 2) build original graphs for this group (NetworkX with BC-int labels)
    originals = {idx: to_nx_with_bc_int(dataset[idx]) for idx in group_indices}

    # 3) create 10 perturbed copies per original
    copies = []
    copies_per_graph = 10
    for idx in group_indices:
        G0 = originals[idx]
        for j in range(copies_per_graph):
            try:
                H, op, e = random_single_edge_perturbation(G0)
                copies.append({"orig_idx": idx, "copy_idx": j, "op": op, "edge": e, "graph": H})
            except ValueError as ve:
                # If a rare graph can't be perturbed, skip this copy
                print(f"Skip idx={idx}, copy={j}: {ve}")

    print(f"\nTotal perturbed copies created: {len(copies)} "
          f"(requested {len(group_indices)*copies_per_graph})")

    # 4) relabel copies with BC-int and compute WL hashes
    copy_hashes = []
    for d in copies:
        H_labeled = relabel_with_bc_int(d["graph"])
        h = weisfeiler_lehman_hash(H_labeled)
        d["wl_hash"] = h
        d["survived"] = (h == group_hash)
        copy_hashes.append(h)

    # 5) aggregate counts
    total = len(copies)
    survived = sum(d["survived"] for d in copies)
    broken = total - survived

    add_total = sum(1 for d in copies if d["op"] == "add")
    add_survived = sum(1 for d in copies if d["op"] == "add" and d["survived"])
    rem_total = sum(1 for d in copies if d["op"] == "remove")
    rem_survived = sum(1 for d in copies if d["op"] == "remove" and d["survived"])

    print("\n--- Experiment A: WL survival after one-edge perturbation ---")
    print(f"Overall copies: {total}")
    print(f"  Survived (same WL hash as original group): {survived}")
    print(f"  Broken   (moved to a different WL hash)  : {broken}")

    print("\nBy operation:")
    print(f"  ADD: survived {add_survived} / {add_total}")
    print(f"  REMOVE: survived {rem_survived} / {rem_total}")

    # (optional) per-original quick view: how many of its 10 copies survived?
    per_orig = {}
    for idx in group_indices:
        c = [d for d in copies if d["orig_idx"] == idx]
        per_orig[idx] = sum(dd["survived"] for dd in c), len(c)
    print("\nPer-original survival (survived/total):")
    for idx in group_indices:
        s, t = per_orig[idx]
        print(f"  graph {idx}: {s}/{t}")