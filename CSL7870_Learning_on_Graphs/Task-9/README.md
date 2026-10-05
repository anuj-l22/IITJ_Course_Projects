# Molecular Graph Classification with Graph2Vec: WL vs. Anonymous Walks

**Course:** CSL7870 – Learning on Graphs
**Author:** Anuj

Graph-level embedding and classification on the **AIDS antiviral screen** dataset, where each molecular graph is labelled as active or inactive against HIV. Molecular graphs were converted into "documents" in two ways, **Weisfeiler–Lehman (WL) refinements** and **Anonymous Walks (AW)**, then embedded with Doc2Vec (PV-DM) and classified with a small MLP.

## Highlights
- **Best macro-F1 0.952** (accuracy 0.962) using Anonymous Walk documents with walk length L = 105 and M = 100 walks.
- The best WL baseline reached **macro-F1 0.944** at 1024 dimensions, with runtime staying under 4 seconds.
- Showed that **longer walks are the efficient lever**. Increasing L raised macro-F1 from 0.42 to 0.95, while adding more short walks only reached 0.56.

## Approach
- **De-duplication:** removed structurally isomorphic graphs using a 3-iteration WL signature seeded with betweenness centrality.
- **WL documents:** labels seeded by node degree and refined iteratively, with each iteration's labels forming one sentence.
- **AW documents:** M random walks of length L, anonymised using visit counts.
- **Sweeps:** WL dimension from 16 to 1024, AW walk length L from 5 to 105, AW walk count M from 10 to 1010, and a coarse L × M grid.
- **Evaluation:** macro and per-class precision, recall and F1, plus confusion matrices and a runtime-vs-quality analysis.

## Results (test set)
| Model | Config | Accuracy | Macro-F1 |
|---|---|---|---|
| WL | d = 128 | 0.9364 | 0.9127 |
| WL (best) | d = 1024 | 0.9576 | 0.9441 |
| **AW (best)** | L = 105, M = 100 | **0.9619** | **0.9520** |
| AW (best on grid) | L = 105, M = 410 | 0.9576 | 0.9453 |

## Key Insights
- Longer anonymous walks capture larger motifs such as rings and functional groups. This boosts recall, especially for the minority class.
- WL is a fast baseline that keeps precision high.
- The jagged macro-precision curve traced back to the small minority class: a swing of ±2 errors visibly shifts the curve.

## Skills & Tools
Graph Representation Learning · Graph2Vec / Doc2Vec · Weisfeiler–Lehman Kernels · Anonymous Walks · Molecular Graphs · Python
