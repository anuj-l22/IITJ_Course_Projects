import collections
import networkx as nx
import numpy as np

def parse_aids_dataset(edge_file, indicator_file, node_labels_file):
    graph_indicator = {}
    with open(indicator_file, 'r') as f:
        for i, line in enumerate(f):
            node_id = i + 1
            graph_id = int(line.strip())
            graph_indicator[node_id] = graph_id

    node_labels = {}
    with open(node_labels_file, 'r') as f:
        for i, line in enumerate(f):
            node_id = i + 1
            label = int(line.strip())
            node_labels[node_id] = label

    num_graphs = max(graph_indicator.values())
    graphs = [nx.Graph() for _ in range(num_graphs)]

    for node_id, graph_id in graph_indicator.items():
        graph_index = graph_id - 1
        label = node_labels.get(node_id)
        if label is not None:
            graphs[graph_index].add_node(node_id, label=label)
        else:
            graphs[graph_index].add_node(node_id)

    with open(edge_file, 'r') as f:
        for line in f:
            u_str, v_str = line.strip().split(',')
            u, v = int(u_str.strip()), int(v_str.strip())

            graph_id = graph_indicator.get(u)
            if graph_id is not None:
                graph_index = graph_id - 1
                graphs[graph_index].add_edge(u, v)

    return graphs

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

def find_isomorphic_groups(graphs):
    hashes = collections.defaultdict(list)
    for i, g in enumerate(graphs):
        if len(g) > 0:
            h = weisfeiler_lehman_hash(g)
            hashes[h].append(i)


    graph2group = {}
    for gid, (_, members) in enumerate(sorted(hashes.items(), key=lambda kv: min(kv[1]))):
        for idx in members:
            graph2group[idx] = gid
    
    y_raw = np.array([graph2group[i] for i in range(len(all_graphs))], dtype=int)
    np.save("labels_raw_bc.npy", y_raw)

    isomorphic_groups = {h: indices for h, indices in hashes.items() if len(indices) > 1}
    return isomorphic_groups

# --- Main Execution ---
if __name__ == "__main__":
    # Define file paths (assuming they are in the same directory)
    EDGE_FILE = 'AIDS/AIDS/AIDS_A.txt'
    INDICATOR_FILE = 'AIDS/AIDS/AIDS_graph_indicator.txt'
    NODE_LABELS_FILE = 'AIDS/AIDS/AIDS_node_labels.txt'

    print("1. Parsing dataset files...")
    try:
        all_graphs = parse_aids_dataset(EDGE_FILE, INDICATOR_FILE, NODE_LABELS_FILE)
        print(f"   Successfully parsed {len(all_graphs)} graphs.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e.filename}")
        print("Please make sure 'AIDS_A.txt', 'AIDS_grapah_indicator.txt', and 'AIDS_node_labels.txt' are in the same directory as the script.")
        exit()
        
    for g in all_graphs:
        if len(g) == 0:
            continue
        bc = nx.betweenness_centrality(g, normalized=False)  # raw counts
        bc_int = {u: int(round(bc[u])) for u in g.nodes()}
        nx.set_node_attributes(g, bc_int, 'label')
        
    print("\n2. Running Weisfeiler-Lehman isomorphism test...")
    isomorphic_groups = find_isomorphic_groups(all_graphs)
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
        # Sort groups by the first graph index in each group for consistent output
        sorted_groups = sorted(isomorphic_groups.values(), key=lambda x: x[0])
        for i, group in enumerate(sorted_groups):
            print(f"     - Group {i+1}: {group}")
