# Sanskrit–English Cross-Lingual Sentence Embeddings

**Course:** AIL7390 – Deep Learning for NLP
**Author:** Anuj

Built compact, semantically aligned sentence embeddings for parallel Sanskrit–English text. Sanskrit is a low-resource, morphologically rich language pair. The goal was to maximise cosine similarity between aligned pairs while keeping the embedding dimension as small as possible.

## Highlights
- **0.9862 average test cosine similarity at only 32 dimensions** with fine-tuned Jina Embeddings v5. This reached the maximum rubric score for both quality and dimensionality.
- Contrastive fine-tuning improved on the zero-shot model by **+0.48 cosine (absolute)**.
- Benchmarked three multilingual embedding models in three settings: zero-shot, contrastive fine-tuning and learned projection.

## Approach
- **Models:** `sanganaka/bge-m3-sanskritFT`, `Qwen/Qwen3-Embedding-0.6B` and `jinaai/jina-embeddings-v5-text-small`.
- **Baselines:** random embeddings and zero-shot inference.
- **Fine-tuning:** symmetric InfoNCE contrastive loss (τ = 0.05) on 10k parallel pairs. Matryoshka truncation produced 32/64/128-d embeddings. Training used bf16 mixed precision and gradient checkpointing.
- **Dimension reduction:** learned linear projection heads trained on frozen encoders.
- **Analysis:** t-SNE of aligned pairs, cosine distributions, best- and worst-aligned pair inspection, and the trade-off between dimension and cosine similarity.

## Results (test set)
| Model | Method | Dim | Cosine |
|---|---|---|---|
| Jina v5 | Fine-tuned (InfoNCE) | 32 | **0.9862** |
| BGE-M3 Sanskrit | Fine-tuned + projection | 32 | 0.8078 |
| Qwen3-Embedding | Fine-tuned (Matryoshka) | 32 | 0.8059 |
| BGE-M3 Sanskrit | Fine-tuned | 1024 | 0.7515 |

## Key Insights
- Models trained natively with Matryoshka representations handle aggressive low-dimensional embedding best.
- Projecting BGE-M3 from 1024-d down to 32-d *improved* alignment (0.7579 → 0.8127 on dev). This shows the full 1024-d space contains redundant dimensions.
- Fine-tuning directly at a low dimension beats projecting afterwards: Qwen fine-tuned at 32-d scored 0.8092, while a 128→32 projection scored 0.7436.
- Code-mixed sentences align almost perfectly. Purely literary Sanskrit is still the hardest case.

## Skills & Tools
PyTorch · Hugging Face Transformers · Sentence-Transformers · scikit-learn · Contrastive Learning · Matryoshka Representation Learning · Cross-lingual NLP
