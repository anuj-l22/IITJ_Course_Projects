# Sanskrit-to-English Neural Machine Translation

**Course:** AIL7390 – Deep Learning for NLP
**Author:** Anuj

Built a neural machine translation system for Sanskrit → English, a low-resource setting that is hard because of sandhi (word fusion), flexible word order and the Devanagari script. Fine-tuned and compared two pretrained sequence-to-sequence Transformers: a subword-based model and a byte-level model. Evaluation covered both translation quality and efficiency.

## Highlights
- **MarianMT: BLEU 0.2014 and BERTScore F1 0.4986 on the test set**, nearly double the BLEU of the byte-level baseline.
- **13× faster inference** (11.4 vs. 0.9 sentences/s) with **less than half the parameters** (140M vs. 301M).
- Analysed the failure modes of each architecture: domain hallucination for the subword model and transliteration for the byte-level model.

## Approach
- **Models:** MarianMT (`Helsinki-NLP/opus-mt-inc-en`, SentencePiece subwords, Indic→English pretraining) and ByT5-small (`google/byt5-small`, tokenizer-free UTF-8 bytes).
- **Data:** 10k / 1k / 1k parallel train / dev / test pairs. Sequence-length limits were set from an analysis of token lengths.
- **Fine-tuning:** HuggingFace `Seq2SeqTrainer` with AdamW, label smoothing (0.1), warmup and early stopping. MarianMT used FP16. ByT5 used FP32 because it was unstable under FP16.
- **Decoding:** beam search ablation over k ∈ {1, 4, 8} with no-repeat-n-gram blocking.
- **Evaluation:** NLTK BLEU, BERTScore (roberta-large, rescaled), inference time and parameter count.

## Results (test set)
| Model | BLEU | BERTScore F1 | Inference (1k sentences) | Params |
|---|---|---|---|---|
| **MarianMT** (beam 8) | **0.2014** | **0.4986** | **87.4 s** | 139.6M |
| ByT5-small (beam 4) | 0.1074 | 0.3732 | 1163.5 s | 300.8M |

## Key Insights
- With only 10k training pairs, task-specific pretraining on Indic translation mattered more than byte-level tokenization.
- Byte sequences are about 3× longer than subword sequences, which multiplies the self-attention cost by roughly 9×.
- BLEU improved steadily with beam width for MarianMT. ByT5 peaked at beam 4.
- Code-mixed sentences inflate scores. Purely literary Sanskrit is where both models struggle.

## Skills & Tools
PyTorch · Hugging Face Transformers · Seq2Seq Fine-tuning · Beam Search · BLEU / BERTScore · Low-resource Machine Translation
