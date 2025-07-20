# План изучения LLM + NLP (Senior, Search/RecSys)

## I. Архитектуры и фундамент
**Цель:** Понимать внутреннее устройство LLM, различия моделей, использовать и модифицировать под задачу.

- Принцип работы Transformer: attention, encoder/decoder схемы  
  🔗 [Transformers illustrated](https://jalammar.github.io/illustrated-transformer/)  
  🔗 [Transformers code](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)  
  🔗 [Attention is all you need](https://arxiv.org/pdf/1706.03762)  
  🔗 [Mixture of Experts Explained](https://huggingface.co/blog/moe)  
  🔗 [A Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)  
  🔗 [Understanding Mixture of Experts: Building a MoE Model with PyTorch](https://medium.com/@prateeksikdar/understanding-mixture-of-experts-building-a-moe-model-with-pytorch-dd373d9db81c)

- Архитектуры: BERT, GPT, T5, LLaMA, Mistral, Claude, Gemini и т.п.  
  🔗 [BERT](https://huggingface.co/blog/bert-101)  
  🔗 [GPT-3](https://dugas.ch/artificial_curiosity/GPT_architecture.html)  
  🔗 [T5](https://medium.com/40gagangupta_82781understanding-the-t5-model-a-comprehensive-guide-b4d5c02c234b)  
  🔗 [LLama3-1](https://ai.meta.com/blog/meta-llama-3-1/)  
  🔗 [LLama4-1](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)  
  🔗 [Mistral 7b](https://medium.com/dair-ai/papers-explained-mistral-7b-b9632dedf580)  
  🔗 [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts)  
  🔗 [Gemini 2.5](https://arxiv.org/pdf/2507.06261)  

- Отличия: causal vs masked LM, decoder-only vs encoder-decoder  
  🔗 https://huggingface.co/course/chapter1  
  🔗 https://medium.com/%40tom_21755/understanding-causal-llms-masked-llm-s-and-seq2seq-a-guide-to-language-model-training-d4457bbd07fa  
  🔗 https://huggingface.co/docs/transformers/en/tasks/language_modeling

- Устройство токенизации (BPE, SentencePiece)  
  🔗 https://huggingface.co/course/chapter6  
  🔗 https://huggingface.co/docs/tokenizers/index

- Слои модели: Embeddings, LayerNorm, FeedForward, Positional Encoding, Attention  
  🔗 https://lilianweng.github.io/lil-log/

- Работа с `transformers`, `config`, `forward`, `past_key_values`  
  🔗 https://huggingface.co/docs/transformers/index

---

## II. Файнтюнинг и адаптация моделей
**Цель:** Уметь адаптировать любую LLM под свою задачу с минимальными затратами.

- Классический fine-tuning (FP32, full finetune)  
  🔗 https://huggingface.co/course/chapter3

- Parameter-efficient tuning:  
  🔗 https://github.com/huggingface/peft  
  🔗 https://arxiv.org/abs/2305.14314 (QLoRA)  
  🔗 https://sebastianraschka.com/blog/2023/peft-qlora.html

- Low-bit inference: quantization (int8, int4), `bitsandbytes`, `AutoGPTQ`  
  🔗 https://github.com/TimDettmers/bitsandbytes  
  🔗 https://github.com/PanQiWei/AutoGPTQ

- Использование `PEFT`, `Trainer`, `accelerate`, `deepspeed`  
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

- Temperature, top_p, repetition_penalty  
  🔗 https://platform.openai.com/docs/guides/text-generation

- Prompt compression, reranking, robustness  
  🔗 https://arxiv.org/abs/2309.02772 (Prompt Compression for LLMs)

- Chain-of-prompt, tree-of-thought, multi-hop reasoning  
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

## V. Retrieval и RAG
**Цель:** Строить пайплайны retrieval-augmented generation и делать LLM по-настоящему полезными.

- Основы RAG  
  🔗 https://arxiv.org/abs/2005.11401  
  🔗 https://www.pinecone.io/learn/retrieval-augmented-generation/  
  🔗 https://www.llamaindex.ai/guides/retrievers/rag-intro

- FAISS, pgvector, Chroma, Pinecone  
  🔗 https://github.com/facebookresearch/faiss  
  🔗 https://github.com/pgvector/pgvector  
  🔗 https://github.com/chroma-core/chroma  
  🔗 https://www.pinecone.io/docs/

- Hybrid search, reranking:  
  🔗 https://zilliz.com/blog/hybrid-search  
  🔗 https://github.com/stanford-futuredata/ColBERT  
  🔗 https://github.com/naver/splade

- RAG evaluation:  
  🔗 https://github.com/explodinggradients/ragas

---

## VI. Оптимизация inference и latency
**Цель:** Разворачивать и использовать модели эффективно и с минимальной задержкой.

- vLLM  
  🔗 https://docs.vllm.ai/

- TGI (Text Generation Inference)  
  🔗 https://huggingface.co/docs/text-generation-inference

- Flash Attention, speculative decoding  
  🔗 https://github.com/Dao-AILab/flash-attention  
  🔗 https://arxiv.org/abs/2302.01318 (Speculative Decoding)

- Quantization-aware training, AutoGPTQ  
  🔗 https://github.com/PanQiWei/AutoGPTQ  
  🔗 https://github.com/huggingface/optimum

---

## VII. Анализ и отладка моделей
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

## VIII. LLM в системах поиска и рекомендаций
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

## IX. Дополнительно (опционально)
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