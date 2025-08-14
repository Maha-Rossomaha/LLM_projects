# План изучения LLM + NLP (Senior, Search/RecSys)

## I. Архитектуры и фундамент
**Цель:** Понимать внутреннее устройство LLM, различия моделей, использовать и модифицировать под задачу.

- **Transformers**  
  🔗 [Transformers illustrated](https://jalammar.github.io/illustrated-transformer/)  
  🔗 [Transformers code](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)  
  🔗 [Attention is all you need](https://arxiv.org/pdf/1706.03762)  
  🔗 [Mixture of Experts Explained](https://huggingface.co/blog/moe)  
  🔗 [A Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)  
  🔗 [Understanding Mixture of Experts: Building a MoE Model with PyTorch](https://medium.com/@prateeksikdar/understanding-mixture-of-experts-building-a-moe-model-with-pytorch-dd373d9db81c)

- **Architectures**  
  🔗 [HF LM](https://huggingface.co/course/chapter1)  
  🔗 [Understanding Causal LLM’s, Masked LLM’s, and Seq2Seq](https://medium.com/%40tom_21755/understanding-causal-llms-masked-llm-s-and-seq2seq-a-guide-to-language-model-training-d4457bbd07fa)  
  🔗 [HF Encoder-Decoder](https://huggingface.co/learn/llm-course/en/chapter1/6)
  
- **Tokenizers**  
  🔗 [HF Tokenizers](https://huggingface.co/course/chapter6)  

- **Embeddings**  
  🔗 [Embedding layer tutorial: A comprehensive guide to neural network representations](https://www.byteplus.com/en/topic/400368)  
  🔗 [Word2Vec](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)  
  🔗 [GloVe](https://aclanthology.org/D14-1162/)  
  🔗 [FastText](https://arxiv.org/abs/1607.04606)

- **Layers and Activations**   
  🔗 [Layer Normalization](https://medium.com/@aisagescribe/ace-ai-interview-series-8-what-is-the-common-normalization-method-in-llm-training-18e559f46e08)  
  🔗 [Feed-Forward](https://sebastianraschka.com/blog/2023/transformer-feedforward.html)  
  🔗 [Positional Encoding](https://codelabsacademy.com/ru/news/roformer-enhanced-transformer-with-rotary-position-embedding-2024-5-30/)  
  🔗 [Dropout](https://habr.com/ru/companies/wunderfund/articles/330814/)  

- **Models**  
  🔗 [BERT](https://huggingface.co/blog/bert-101)  
  🔗 [GPT-3](https://dugas.ch/artificial_curiosity/GPT_architecture.html)  
  🔗 [T5](https://medium.com/40gagangupta_82781understanding-the-t5-model-a-comprehensive-guide-b4d5c02c234b)  
  🔗 [LLama3-1](https://ai.meta.com/blog/meta-llama-3-1/)  
  🔗 [LLama4-1](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)  
  🔗 [Mistral 7b](https://medium.com/dair-ai/papers-explained-mistral-7b-b9632dedf580)  
  🔗 [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts)  
  🔗 [Gemini 2.5](https://arxiv.org/pdf/2507.06261)  

---

## II. Файнтюнинг и адаптация моделей
**Цель:** Уметь адаптировать любую LLM под свою задачу с минимальными затратами.

- **Classic fine-tuning (FP32, full finetune)**  
  🔗 [Fine Tuning](https://huggingface.co/course/chapter3)  

- **Parameter-efficient tuning**  
  🔗 [PEFT](https://habr.com/ru/articles/791966/)  
  🔗 [LoRA](https://arxiv.org/abs/2106.09685)  
  🔗 [QLoRA 1](https://medium.com/@gitlostmurali/understanding-lora-and-qlora-the-powerhouses-of-efficient-finetuning-in-large-language-models-7ac1adf6c0cf)  
  🔗 [QLoRA 2](https://www.unite.ai/lora-qlora-and-qa-lora-efficient-adaptability-in-large-language-models-through-low-rank-matrix-factorization/)  
  🔗 [QLoRA 3](https://sebastianraschka.com/blog/2023/peft-qlora.html)  
  🔗 [Prefix Tuning](https://arxiv.org/abs/2101.00190)  
  🔗 [Prompt Tuning](https://arxiv.org/abs/2104.08691)  
  🔗 [P-tuning v2](https://arxiv.org/abs/2110.07602)  
  🔗 [Adapters](https://magazine.sebastianraschka.com/p/finetuning-llms-with-adapters#:~:text=The%20idea%20of%20parameter%2Defficient,the%20pretrained%20LLM%20remain%20frozen.)  
  🔗 [Domain Adaptation and Comparison]()

- **Alignment**  
  🔗 [InstuctGPT](https://arxiv.org/abs/2203.02155)  
  🔗 [RLHF](https://huggingface.co/blog/rlhf)  
  🔗 [DPO](https://huggingface.co/blog/pref-tuning)  

- **Low-bit inference: quantization**  
  🔗 [A Survey on Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2003.13630)  

- **Effective Training**:  
  🔗 [Gradient Accumulation and Checkpointing](https://aman.ai/primers/ai/grad-accum-checkpoint/)  
  🔗 [Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)  
  🔗 [Pipeline Parallelism](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)  
  🔗 [Tensor Parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)  
  🔗 [ZeRO stages](https://huggingface.co/docs/accelerate/v0.10.0/en/deepspeed)  

- **Instruments**  
  🔗 [PEFT](https://huggingface.co/docs/peft/index)  
  🔗 [Accelerate](https://github.com/huggingface/accelerate)  
  🔗 [DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)

- **Long context**  
  🔗 [NTK Scaling](https://en.wikipedia.org/wiki/Neural_tangent_kernel)  
  🔗 [Flash Attention](https://github.com/Dao-AILab/flash-attention)  

---

## III. Prompt Engineering и контроль генерации
**Цель:** Проектировать промпты под любые задачи, снижать галлюцинации, обеспечивать стабильность генерации.

- **In-context learning**   
  🔗 [Zero, One and Few-Shot](https://arxiv.org/abs/2301.00234)  
  🔗 [Chain of Thought](https://arxiv.org/abs/2201.11903)  
  🔗 [SelfAsk](https://arxiv.org/abs/2210.03350)  
  🔗 [ReAct](https://arxiv.org/abs/2210.03629)  

- **Decoding**  
  🔗 [Decoding Parameters](https://platform.openai.com/docs/guides/text-generation)  
  🔗 [Decoding Algorithms](http://arxiv.org/html/2402.06925v3)  

- **Reasoning**  
  🔗 [Tree of Thought](https://arxiv.org/abs/2305.10601)  
  🔗 [Multi-hop Reasoning](https://www.moveworks.com/us/en/resources/ai-terms-glossary/multi-hop-reasoning)

- **Prompt evaluation**  
  🔗 [Promptfoo](https://github.com/promptfoo/promptfoo)  
  🔗 [OpenAI Evals](https://github.com/open-eval/open-eval)

---

## IV. Embeddings и векторные представления
**Цель:** Строить и использовать dense-представления для поиска, рекомендаций, кластеризации, дедупликации.

- **Text Vector Representations**   
  🔗 [Dense and Sparse Embeddings](https://mlokhandwalas.medium.com/dense-and-sparse-embeddings-a-comprehensive-overview-c5f6473ee9d0)  
  🔗 [Contextual Embeddings](https://arxiv.org/abs/2003.07278)  
  🔗 [Sentence Embeddings](https://cohere.com/llmu/sentence-word-embeddings)  

- **Adaptation** Triplet loss, contrastive learning  
  🔗 [Contrastive Learning](https://medium.com/@sulbha.jindal/new-llm-learning-method-contrastive-learning-19425fda59a6)  
  🔗 [Triplet Loss](https://www.v7labs.com/blog/triplet-loss)  
  🔗 [Info-NCE](https://arxiv.org/pdf/2402.05369)  
  🔗 [Supervised Contrastive Loss](https://arxiv.org/abs/2004.11362)  
  🔗 [Negatives Mining](https://arxiv.org/pdf/2407.15831)  
  🔗 [Hard and Soft Negatives]()  
  🔗 [ACNE: Asymmetric Contrastive Negative Example Mining]()  
  🔗 [MoCo and Memory Bank](https://arxiv.org/html/2501.16360v1)  
  🔗 [Inductive Bias]()  



- **Clustering**  
  🔗 [Clustering](https://scikit-learn.org/stable/modules/clustering.html)  
  🔗 [Text Clustering Algorithms and Metrics](https://arxiv.org/html/2403.15112v5)  
  🔗 [UMAP](https://umap-learn.readthedocs.io/en/latest/)  
  🔗 [DAPT and TAPT](https://ceur-ws.org/Vol-2723/short33.pdf)  
  🔗 [Inductive Bias](https://arxiv.org/html/2402.18426v1)  

- **Embedding Models**  
  🔗 [GTE](https://arxiv.org/abs/2308.03281)  
  🔗 [BGE](https://arxiv.org/abs/2402.03216)  
  🔗 [E5](https://arxiv.org/abs/2212.03533)  
  🔗 [MiniLM](https://arxiv.org/abs/2002.10957)  
  🔗 [Cohere Embed]()  
  🔗 [Ada](https://arxiv.org/abs/2401.12421)  
  🔗 [SBERT](https://arxiv.org/abs/1908.10084)  

---

## V. Анализ и отладка моделей
**Цель:** Понимать поведение модели, отлавливать ошибки, снижать токсичность и галлюцинации.

- **Model Interpretation**  
  🔗 [Attention tracing and BertViz](https://medium.com/@GaryFr0sty/visualize-attention-scores-of-llms-with-bertviz-3deb94b455b3)    
  🔗 [Token-Level Logit Analysis](https://arxiv.org/abs/1706.04599)  
  🔗 [Layer-Wise Relevance Propagation](https://arxiv.org/abs/1509.06321)  
  🔗 [Integrated Gradients](https://arxiv.org/abs/1703.01365)  
  🔗 [SHAP GitHub](https://github.com/shap/shap)  
  🔗 [Captum (PyTorch Explainability)](https://captum.ai/)

- **Diagnosis of Errors and Hallucinations**  
  🔗 [Hallucination Sources](https://medium.com/@tam.tamanna18/understanding-llm-hallucinations-causes-detection-prevention-and-ethical-concerns-914bc89128d0)  
  🔗 [Faithfulness-tests](https://arxiv.org/abs/2305.18029)  
  🔗 [Toxicity Bias Tests](https://medium.com/@rajneeshjha9s/tools-to-identify-and-mitigate-bias-toxicity-in-llms-b34e95732241)

- **Тестирование устойчивости (robustness)**  
  🔗 [Adversarial Prompting](https://www.promptingguide.ai/risks/adversarial)  
  🔗 [Prompt Mutations](https://elsworth.phd/Formalisms/A-Survey-of-Prompt-Mutations)  
  🔗 [Stress Tests Long Inputs](https://arxiv.org/abs/2307.03172)

---

## VI. LLM в системах поиска и рекомендаций
**Цель:** Применять всё вышеописанное в end-to-end пайплайнах.

- **Semantic Search**   
  🔗 [Semantic Search](https://www.pinecone.io/learn/semantic-search/)  
  🔗 [Retrieval Reranking](https://sebastianraschka.com/blog/2023/retrieval-reranking.html)  
  🔗 [Hybrid Search](https://www.trychroma.com/docs/hybrid-search)

- **Answer generation**  
  🔗 [LangChain Generation](https://github.com/langchain-ai/langchain/blob/master/cookbook/search_rag.md)


---

## VII. Дополнительно (опционально)
- Сравнение моделей  
  🔗 https://paperswithcode.com/llm-leaderboard  
  🔗 https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

- Контекстная компрессия  
  🔗 https://arxiv.org/abs/2309.02772  
  🔗 https://arxiv.org/abs/2307.03172 (Selectively forgetting with token pruning)

- Безопасность и приватность  
  🔗 https://arxiv.org/abs/2306.15595 (PII Risk in LLMs)

- Датасеты и аннотация  
  🔗 https://huggingface.co/docs/datasets/index  
  🔗 https://github.com/facebookresearch/dynabench

---

  ## VIII. Дорожная карта компетенций (12 недель)

| Недели | Ключевая цель | Результат / метрика |
|--------|---------------|---------------------|
| **1 – 2** | Разобрать математику self-attention и реализовать mini-Transformer «с нуля» (PyTorch) | Ноутбук с кодом + объяснение, почему сложность $O(n^2)$ и как её снижает FlashAttention |
| **3 – 4** | Fine-tune *BERT-base* на SQuAD v1 (sub-set) | F1 ≥ 80 % и отчёт о гиперпараметрах |
| **5 – 6** | LoRA / QLoRA-адаптация *Llama-3-8B* под свой доменный корпус | Δ perplexity ≥ -15 % при GPU memory ≤ 12 GB |
| **7 – 8** | Отладить prompt-паттерны (CoT, ReAct) и снизить hallucination-rate | ≈ 50 ручных кейсов → доля галлюцинаций ≤ 10 % |
| **9 – 10** | Собрать гибридный поиск (*BM25 + E5-small*) и измерить качество | nDCG@10 ≥ 0.40 на треке MTEB-QA |
| **11 – 12** | Запустить RAG-прототип (retriever → reranker → generator) + автоматический evaluation | groundedness ≥ 0.60, regression guardrails в CI |

> **Параллельно:** вести журнал экспериментов, логировать метрики (lm-eval-harness, ragas) и сохранять репо с чистыми README.
