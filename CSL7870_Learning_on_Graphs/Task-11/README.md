# GraphSAGE vs. GAT for Node Classification on Cora

**Course:** CSL7870 – Learning on Graphs
**Author:** Anuj

Implemented and compared **GraphSAGE** (with mean, max and LSTM aggregators) and **Graph Attention Networks** (with 1, 2 and 5 heads) for semi-supervised node classification on the Cora citation network. The study covers full 1,433-dimensional features and compressed autoencoder features (64, 128 and 256 dimensions). It also includes a t-SNE analysis of oversmoothing in deep GNNs.

## Highlights
- **Best test accuracy 81.4%**, from a 1-head GAT that used **about half the parameters** of GraphSAGE-mean (92K vs. 184K).
- Used an autoencoder to compress features, which made the LSTM aggregator trainable without running out of memory.
- A 2-head GAT stayed robust under compression (78.7–79.7%), while GraphSAGE dropped to 75–76%.
- Visualised **oversmoothing**: a 19-layer GAT loses the clear class clusters that the 2-layer model keeps.

## Approach
- **Dataset:** Cora, with 2,708 nodes, 1,433 features and 7 classes, using the standard Planetoid split.
- **Models:** 2-layer GraphSAGE (mean, max and LSTM aggregators) and 2-layer GAT (H ∈ {1, 2, 5} heads).
- **Training:** hidden size 64, dropout 0.5, Adam (lr 0.01, weight decay 5e-4), 200 epochs. Test accuracy was reported at the epoch with the best validation accuracy.
- **Analysis:** per-class precision, recall and F1, confusion matrices, loss and accuracy curves, and t-SNE of the learned embeddings.

## Results (test accuracy %)
| Model | 1433-D | 64-D | 128-D | 256-D |
|---|---|---|---|---|
| GraphSAGE – mean | 80.8 | 75.9 | 75.6 | 74.8 |
| GraphSAGE – max | 78.5 | 73.1 | 71.3 | 74.4 |
| GraphSAGE – LSTM | OOM | 58.4 | 55.0 | 57.1 |
| GAT – 1 head | **81.4** | 75.3 | 79.1 | 79.3 |
| GAT – 2 heads | **81.4** | **78.7** | **79.4** | **79.7** |
| GAT – 5 heads | 80.3 | 77.0 | 76.6 | 79.3 |

## Key Insights
- A GAT with 1–2 heads is a strong, parameter-efficient default on Cora.
- The LSTM aggregator underperforms despite having far more parameters.
- The most common error across all settings is confusing class 3 with class 4.

## Skills & Tools
Graph Neural Networks · GraphSAGE · Graph Attention Networks · Autoencoders · t-SNE · Node Classification
