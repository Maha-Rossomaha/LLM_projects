# План изучения LLM + NLP (Senior, Search/RecSys)

## I. Архитектуры и фундамент
**Цель:** Понимать внутреннее устройство LLM, различия моделей, использовать и модифицировать под задачу.

- **Transformers**:  
  🔗 [Transformers illustrated](https://jalammar.github.io/illustrated-transformer/)  
  🔗 [Transformers code](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)  
  🔗 [Attention is all you need](https://arxiv.org/pdf/1706.03762)  
  🔗 [Mixture of Experts Explained](https://huggingface.co/blog/moe)  
  🔗 [A Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)  
  🔗 [Understanding Mixture of Experts: Building a MoE Model with PyTorch](https://medium.com/@prateeksikdar/understanding-mixture-of-experts-building-a-moe-model-with-pytorch-dd373d9db81c)

- **Architectures**:  
  🔗 [HF LM](https://huggingface.co/course/chapter1)  
  🔗 [Understanding Causal LLM’s, Masked LLM’s, and Seq2Seq](https://medium.com/%40tom_21755/understanding-causal-llms-masked-llm-s-and-seq2seq-a-guide-to-language-model-training-d4457bbd07fa)  
  🔗 [HF Encoder-Decoder](https://huggingface.co/learn/llm-course/en/chapter1/6)
  
- **Tokenizers**  
  🔗 [HF Tokenizers](https://huggingface.co/course/chapter6)  

- **Embeddings**  
  🔗 [Embedding layer tutorial: A comprehensive guide to neural network representations](https://www.byteplus.com/en/topic/400368)  
  🔗 [Word2Vec](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)  
  🔗 [GloVe](https://aclanthology.org/D14-1162/)

- **Layers and Activations**:   
  🔗 Layer Normalization  
  🔗 [Feed-Forward](https://sebastianraschka.com/blog/2023/transformer-feedforward.html)  
  🔗 [Positional Encoding](https://codelabsacademy.com/ru/news/roformer-enhanced-transformer-with-rotary-position-embedding-2024-5-30/)  
  🔗 [Dropout](https://habr.com/ru/companies/wunderfund/articles/330814/)  

- **Models**:  
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

- **Классический fine-tuning (FP32, full finetune)**  
  🔗 https://huggingface.co/course/chapter3

- **Parameter-efficient tuning**  
  🔗 [PEFT](https://habr.com/ru/articles/791966/)  
  🔗 [LoRA](https://arxiv.org/abs/2106.09685)  
  🔗 [QLoRA 1](https://medium.com/@gitlostmurali/understanding-lora-and-qlora-the-powerhouses-of-efficient-finetuning-in-large-language-models-7ac1adf6c0cf)  
  🔗 [QLoRA 2](https://www.unite.ai/lora-qlora-and-qa-lora-efficient-adaptability-in-large-language-models-through-low-rank-matrix-factorization/)  
  🔗 [QLoRA 3](https://sebastianraschka.com/blog/2023/peft-qlora.html)  
  🔗 [Prefix Tuning](https://arxiv.org/abs/2101.00190)  
  🔗 [Prompt Tuning](https://arxiv.org/abs/2104.08691)  
  🔗 [Adapters](https://magazine.sebastianraschka.com/p/finetuning-llms-with-adapters#:~:text=The%20idea%20of%20parameter%2Defficient,the%20pretrained%20LLM%20remain%20frozen.)  

- Low-bit inference: quantization (int8, int4), `bitsandbytes`, `AutoGPTQ`  
  🔗 https://github.com/TimDettmers/bitsandbytes  
  🔗 https://github.com/PanQiWei/AutoGPTQ

- Инструменты `PEFT`, `Trainer`, `accelerate`, `deepspeed`  
  🔗 https://github.com/huggingface/accelerate  
  🔗 https://huggingface.co/docs/transformers/perf_train_gpu_one

- Поддержка длинного контекста (NTK scaling, FlashAttention, LongContext)  
  🔗 https://github.com/Dao-AILab/flash-attention  
  🔗 https://huggingface.co/LongChat  
  🔗 https://blog.llamaindex.ai/long-context-llms/

---

## III. Prompt Engineering и контроль генерации
**Цель:** Проектировать промпты под любые задачи, снижать галлюцинации, обеспечивать стабильность генерации.

- Few-shot, zero-shot, CoT, ReAct, Self-Ask  
  🔗 https://github.com/dair-ai/Prompt-Engineering-Guide  
  🔗 https://github.com/openai/openai-cookbook

- Параметры decoding: temperature, top_p, repetition_penalty  
  🔗 https://platform.openai.com/docs/guides/text-generation

- Prompt compression, reranking, robustness  
  🔗 https://arxiv.org/abs/2309.02772 (Prompt Compression for LLMs)

- Reasoning стратегии: Chain-of-prompt, tree-of-thought, multi-hop reasoning  
  🔗 https://arxiv.org/abs/2305.10601 (Tree of Thought)  
  🔗 https://github.com/kyegomez/tree-of-thoughts

- Prompt evaluation  
  🔗 https://github.com/promptfoo/promptfoo  
  🔗 https://github.com/open-eval/open-eval

---

## IV. Embeddings и векторные представления
**Цель:** Строить и использовать dense-представления для поиска, рекомендаций, кластеризации, дедупликации.

- Sentence embeddings, contextual embeddings, dense vs sparse  
  🔗 https://www.sbert.net/

- Использование моделей: GTE, BGE, E5, MiniLM, Cohere Embed, Ada  
  🔗 https://huggingface.co/intfloat/e5-large-v2  
  🔗 https://huggingface.co/BAAI/bge-base-en  
  🔗 https://cohere.com/docs/embed

- Triplet loss, contrastive learning  
  🔗 https://www.pinecone.io/learn/series/fine-tune-llm/contrastive-learning/

- Метрики расстояний, кластеризация  
  🔗 https://scikit-learn.org/stable/modules/clustering.html  
  🔗 https://umap-learn.readthedocs.io/en/latest/

---

## V. Анализ и отладка моделей
**Цель:** Понимать поведение модели, отлавливать ошибки, снижать токсичность и галлюцинации.

- LM Evaluation  
  🔗 https://github.com/EleutherAI/lm-evaluation-harness  
  🔗 https://github.com/open-eval/open-eval

- RAG evaluation  
  🔗 https://github.com/explodinggradients/ragas  
  🔗 https://github.com/facebookresearch/RA-Eval

- Attention tracing, token-level logit analysis  
  🔗 https://github.com/cdpierse/transformers-interpret  
  🔗 https://github.com/jessevig/bertviz

- Adversarial prompting, hallucination reduction  
  🔗 https://github.com/thunlp/OpenPrompt  
  🔗 https://arxiv.org/abs/2305.11738 (Faithfulness Benchmarks)

---

## VI. LLM в системах поиска и рекомендаций
**Цель:** Применять всё вышеописанное в end-to-end пайплайнах.

- Semantic Search (DenseRetriever, reranker)  
  🔗 https://www.pinecone.io/learn/semantic-search/  
  🔗 https://sebastianraschka.com/blog/2023/retrieval-reranking.html

- Hybrid Search  
  🔗 https://www.trychroma.com/docs/hybrid-search

- LangChain search example  
  🔗 https://github.com/langchain-ai/langchain/blob/master/cookbook/search_rag.md

- MTEB leaderboard  
  🔗 https://huggingface.co/spaces/mteb/leaderboard

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
