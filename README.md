# IIT Jodhpur: Course Projects

**Author:** Anuj

A collection of machine learning and AI projects completed during coursework at IIT Jodhpur. Topics include foundation models, generative AI, NLP, graph learning, probabilistic ML, reinforcement learning and model interpretability. Each project folder has its own README summarising the problem, the approach and the key results.

## Interpretability & Applied ML
| Project | Course | Summary |
|---|---|---|
| [Interpreting Deepfake Detection on SID-Set](MAP7020_Applied_ML_Lab/) | MAP7020 – Applied ML Lab | Frozen CLIP/DINOv2/SigLIP baselines (90.0% macro-F1) plus sparse-autoencoder analysis of SIDA-7B, with causal feature steering and adversarial vulnerability mapping |

## Foundation Models & Generative AI
| Project | Course | Summary |
|---|---|---|
| [ControlNet from Scratch](AIL7490_MMLVL/) | AIL7490 – MMLVL | Built ControlNet's zero convolutions, trainable encoder copy and hint encoder from scratch on a frozen Stable Diffusion 1.5 |
| [Multi-Agent Paper Reviewer with MCP](CSL7860_Foundation_Models_GenAI/B22AI061_FM_A2/Q2_MAS/) | CSL7860 – Foundation Models and Generative AI | Three LLM agents running as MCP servers; 100% success rate on the evaluation harness with zero constraint violations |
| [SISA Machine Unlearning](CSL7860_Foundation_Models_GenAI/B22AI061_FM_A2/Q1_SISA/) | CSL7860 – Foundation Models and Generative AI | Sharded, sliced retraining that honours data deletions while saving 75–83% of retraining time |
| [MambaVision Analysis & LLM Robustness](CSL7860_Foundation_Models_GenAI/MinorExam_AnujRajanLalla_MambaVision/) | CSL7860 – Foundation Models and Generative AI | Hybrid Mamba-Transformer backbone on ImageNet-R vs. ViT/ConvNeXt, plus LLM tests on noisy prompts and needle-in-a-haystack retrieval |
| [Generative Models: Creation & Evaluation](CSL7860_Foundation_Models_GenAI/B22AI061_hw_foundation_models/) | CSL7860 – Foundation Models and Generative AI | ControlNet optical illusions and MusicGen audio, plus a custom IllusionScore metric |

## Natural Language Processing
| Project | Course | Summary |
|---|---|---|
| [Sanskrit–English Sentence Embeddings](AIL7390_DL4NLP/B22AI061_DL4NLP_1/) | AIL7390 – Deep Learning for NLP | Contrastive fine-tuning reached 0.986 cross-lingual cosine similarity at only 32 dimensions |
| [Sanskrit→English Machine Translation](AIL7390_DL4NLP/B22AI061_DL4NLP_2/) | AIL7390 – Deep Learning for NLP | Fine-tuned MarianMT vs. ByT5: BLEU 0.20 and 13× faster inference |

## Learning on Graphs
| Project | Course | Summary |
|---|---|---|
| [GraphSAGE vs. GAT on Cora](CSL7870_Learning_on_Graphs/Task-11/) | CSL7870 – Learning on Graphs | Compared aggregators and attention heads with full and autoencoder-compressed features (81.4% test accuracy), and analysed oversmoothing |
| [Graph2Vec: WL vs. Anonymous Walks](CSL7870_Learning_on_Graphs/Task-9/) | CSL7870 – Learning on Graphs | Molecular graph classification on the AIDS dataset, reaching 0.952 macro-F1 |
| [Message-Passing Receptive Fields](CSL7870_Learning_on_Graphs/Task-10/) | CSL7870 – Learning on Graphs | Analysis of how GNN receptive fields grow and mix communities on Zachary's Karate Club |

## Advanced Machine Learning
| Project | Course | Summary |
|---|---|---|
| [Gradient Boosted Trees from Scratch](CSL7530_Advanced_ML/B22AI061_B22AI003_AML_Assignment_1/) | CSL7530 – Advanced Machine Learning | ESL Algorithms 10.3 and 10.4 implemented in NumPy: 97.4% accuracy on Digits and robust losses under outliers |
| [Gaussian Processes from Scratch](CSL7530_Advanced_ML/B22AI061_B22AI003_B22AI065_AML_Assignment_2%20(1)/) | CSL7530 – Advanced Machine Learning | GP regression and Laplace GP classification (R² 0.92 on Friedman #1; 98.3% on Two Moons) |
| [VAE & Conditional VAE](CSL7530_Advanced_ML/B22AI061_B22AI003_B22AI065_AML_Assignment_3/) | CSL7530 – Advanced Machine Learning | Generative modelling on MNIST, with label-controlled generation and latent-space analysis |
| [PPO from Scratch](CSL7530_Advanced_ML/B22AI061_B22AI003_B22AI065_AML_Assignment_4/) | CSL7530 – Advanced Machine Learning | Clipped vs. unclipped PPO on CartPole-v1, showing the trade-off between stability and peak return |
