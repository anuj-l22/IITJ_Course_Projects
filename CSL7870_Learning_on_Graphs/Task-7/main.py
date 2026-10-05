import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import collections
import numpy as np

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
        g = to_networkx(data, node_attrs=['x'])

        nx.set_node_attributes(g, {j: int(x[0]) for j, x in enumerate(data.x)}, 'label')

        if len(g) > 0:
            h = weisfeiler_lehman_hash(g)
            hashes[h].append(i)
            
    graph2group = {}
    for gid, (_, members) in enumerate(sorted(hashes.items(), key=lambda kv: min(kv[1]))):
        for idx in members:
            graph2group[idx] = gid
    
    y_main = np.array([graph2group[i] for i in range(len(dataset))], dtype=int)
    np.save("labels_main.npy", y_main)
    print("[Q1] Saved labels_main.npy for ARI/NMI comparison.")
    
    isomorphic_groups = {h: indices for h, indices in hashes.items() if len(indices) > 1}
    return isomorphic_groups

# --- Main Execution ---
if __name__ == "__main__":
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
