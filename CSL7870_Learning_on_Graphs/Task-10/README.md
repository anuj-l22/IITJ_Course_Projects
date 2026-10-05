# Message-Passing Receptive Fields on Zachary's Karate Club

**Course:** CSL7870 – Learning on Graphs
**Author:** Anuj

Visualised and measured the **node-wise receptive fields** (message-passing trees) that a GNN builds on Zachary's Karate Club graph, and studied how they grow with depth. For each of the 34 nodes, the analysis extracted a depth-k BFS tree and the corresponding induced subgraph. Nodes were coloured by club membership to show how quickly information from the opposite community reaches each node.

## Highlights
- **33 of 34 nodes reach the opposite community within 2 hops.** A 2-layer GNN therefore already mixes cross-class signals for almost every node.
- Identified a few **"gateway" nodes** that carry most cross-community message flow.
- Found **structurally equivalent leaf nodes** (Jaccard similarity = 1.0) that become indistinguishable after 2 layers of message passing unless positional or structural encodings are added.

## Approach
- **Depth-2 analysis for all 34 roots:** tree size, induced subgraph size, cross-edges pruned by BFS, triangles and clustering.
- **Class-aware metrics:** the nearest hop at which the opposite class appears, a boundary index, opposite-class fractions at 1 and 2 hops, and gateway intermediates.
- **Depth-growth study (3 vs. 4 hops):** measured coverage and redundancy for central and peripheral roots.

## Key Insights
- Central nodes have large, redundant receptive fields and mix information fastest. Peripheral nodes stay localised at 2 hops.
- The Officer faction is slightly more exposed than Mr. Hi's (mean nearest-opposite hop 1.59 vs. 1.71).
- Once a node's coverage reaches about 97%, extra hops mainly add redundant cycles rather than new information. This signals the risk of oversmoothing in deeper GNNs.

## Skills & Tools
Graph Neural Networks · Message Passing · Graph Analysis · BFS Tree Extraction · Network Visualisation · Python
