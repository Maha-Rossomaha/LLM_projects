# План изучения LLM + NLP (Senior, Search/RecSys)

## I. Архитектуры и фундамент

- **Transformers**  
  📄 [Transformers basics](1%20Theory/1%20Transformers%20Basics/1%20Transformers/1%20Transformers%20basics.md)  
  📄 [Transformers code](1%20Theory/1%20Transformers%20Basics/1%20Transformers/2%20Transformers%20code.md)  
  📄 [Attention is all you need](1%20Theory/1%20Transformers%20Basics/1%20Transformers/3%20Attention%20is%20all%20you%20need.md)  
  📄 [GQA and SWA](1%20Theory/1%20Transformers%20Basics/1%20Transformers/4%20GQA%20and%20SWA.md)  
  📄 [Mixture of Experts](1%20Theory/1%20Transformers%20Basics/1%20Transformers/5%20Mixture%20of%20Experts.md)

- **Architectures**  
  📄 [Causal and Masked LM](1%20Theory/1%20Transformers%20Basics/2%20Architectures/1%20Causal%20and%20Masked%20LM.md)  
  📄 [Encoder-Decoder](1%20Theory/1%20Transformers%20Basics/2%20Architectures/2%20Encoder-Decoder.md)

- **Tokenizers**  
  📄 [Tokenizer basics](1%20Theory/1%20Transformers%20Basics/3%20Tokenizers/1%20Tokenizer.md)  
  📄 [Byte-Pair Encoding](1%20Theory/1%20Transformers%20Basics/3%20Tokenizers/2%20Byte-Pair%20Encoding.md)  
  📄 [WordPiece](1%20Theory/1%20Transformers%20Basics/3%20Tokenizers/3%20WordPiece.md)  
  📄 [Unigram Language Model](1%20Theory/1%20Transformers%20Basics/3%20Tokenizers/4%20Unigram%20Language%20Model.md)  
  📄 [Byte level BPE](1%20Theory/1%20Transformers%20Basics/3%20Tokenizers/5%20Byte%20level%20BPE.md)  
  📄 [SentencePiece](1%20Theory/1%20Transformers%20Basics/3%20Tokenizers/6%20SentencePiece.md)

- **Embeddings**  
  📄 [Embeddings](1%20Theory/1%20Transformers%20Basics/4%20Embeddings/1%20Embeddings.md)  
  📄 [Word2Vec](1%20Theory/1%20Transformers%20Basics/4%20Embeddings/2%20Word2Vec.md)  
  📄 [GloVe](1%20Theory/1%20Transformers%20Basics/4%20Embeddings/3%20GloVe.md)  

- **Layers and Activations**  
  📄 [Positional Encoding](1%20Theory/1%20Transformers%20Basics/5%20Layers%20and%20Activations/1%20Positional%20Encoding.md)  
  📄 [Normalization](1%20Theory/1%20Transformers%20Basics/5%20Layers%20and%20Activations/2%20Normalization.md)  
  📄 [Dropout](1%20Theory/1%20Transformers%20Basics/5%20Layers%20and%20Activations/3%20Dropout.md)  
  📄 [Feed Forward](1%20Theory/1%20Transformers%20Basics/5%20Layers%20and%20Activations/4%20Feed%20Forward.md)  
  📄 [Activations](1%20Theory/1%20Transformers%20Basics/5%20Layers%20and%20Activations/5%20Activations.md)  
  📄 [Residual Connection](1%20Theory/1%20Transformers%20Basics/5%20Layers%20and%20Activations/6%20Residual%20Connection.md)

- **Models**  
  📄 [BERT](1%20Theory/1%20Transformers%20Basics/6%20Models/1%20BERT.md)  
  📄 [GPT-3](1%20Theory/1%20Transformers%20Basics/6%20Models/2%20GPT-3.md)  
  📄 [T5](1%20Theory/1%20Transformers%20Basics/6%20Models/3%20T5%20Model.md)  
  📄 [LLama3.1](1%20Theory/1%20Transformers%20Basics/6%20Models/4%20LLama3_1.md)  
  📄 [LLama4.1](1%20Theory/1%20Transformers%20Basics/6%20Models/5%20LLama4_1.md)  
  📄 [Mistral 7B](1%20Theory/1%20Transformers%20Basics/6%20Models/6%20Mistral%207B.md)  
  📄 [Mixtral 8x7B](1%20Theory/1%20Transformers%20Basics/6%20Models/7%20Mixtral%C2%A08x7B.md)  
  📄 [Claude](1%20Theory/1%20Transformers%20Basics/6%20Models/8%20Claude.md)  
  📄 [Gemini 2.5](1%20Theory/1%20Transformers%20Basics/6%20Models/9%20Gemini%202_5.md)

---

## II. Файнтюнинг и адаптация моделей  

- **Classic fine-tuning (FP32, full finetune)**  
  📄 [Classic Finetuning](1%20Theory/2%20Finetuning%20and%20Adaptation/1%20Classic%20Finetuning/1%20Classic%20Finetuning.md)  

- **Parameter-efficient tuning**  
  📄 [PEFT](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/1%20PEFT.md)  
  📄 [LoRA](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/2%20LoRA.md)  
  📄 [QLoRA](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/3%20QLoRA.md)  
  📄 [Prompt Tuning](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/4%20Prompt%20Tuning.md)  
  📄 [Prefix Tuning](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/5%20Prefix%20Tuning.md)  
  📄 [P-Tuning v2](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/6%20P-Tuning.md)  
  📄 [Prompt, Prefix and P-Tunings comparison](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/7%20Comparison%20of%20Prompt,%20Prefix%20and%20P%20Tunings.md)  
  📄 [LoRA vs Tunings](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/8%20LoRA%20vs%20Tunings.md)  
  📄 [Adapters](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/9%20Adapters.md)  
  📄 [Domain Adaptation](1%20Theory/2%20Finetuning%20and%20Adaptation/2%20Parameter%20Efficient%20Tuning/10%20Domain%20Adaptation.md)  

- **Alignment**  
  📄 [Alignment](1%20Theory/2%20Finetuning%20and%20Adaptation/3%20Alignment/1%20Alignment.md)  
  📄 [Methods Comparison (PPO, DPO, KTO)](1%20Theory/2%20Finetuning%20and%20Adaptation/3%20Alignment/2%20Methods%20Comparison.md)  

- **Low-bit inference: quantization**  
  📄 [Codebooks](1%20Theory/2%20Finetuning%20and%20Adaptation/4%20Quantization/0%20Codebooks.md)  
  📄 [Review and Motivation](1%20Theory/2%20Finetuning%20and%20Adaptation/4%20Quantization/1%20%20Review%20and%20Motivation.md)  
  📄 [Quantization Types](1%20Theory/2%20Finetuning%20and%20Adaptation/4%20Quantization/2%20Quantization%20Types.md)  
  📄 [Quantization Algorithms](1%20Theory/2%20Finetuning%20and%20Adaptation/4%20Quantization/3%20Quantization%20Algorithms.md)  
  📄 [Quantization Frameworks](1%20Theory/2%20Finetuning%20and%20Adaptation/4%20Quantization/4%20Quantization%20Frameworks.md)

- **Effective Training**  
  📄 [Memory Usage LLM](1%20Theory/2%20Finetuning%20and%20Adaptation/5%20Effective%20Training/1%20Memory%20Usage%20Llm.md)  
  📄 [Gradient Accumulation and Checkpointing](1%20Theory/2%20Finetuning%20and%20Adaptation/5%20Effective%20Training/2%20Gradient%20Accumulation%20%D0%B8%20Checkpointing.md)  
  📄 [Distributed Data Parallel](1%20Theory/2%20Finetuning%20and%20Adaptation/5%20Effective%20Training/3%20Distributed%20Data%20Parallel.md)  
  📄 [Pipeline Parallelism](1%20Theory/2%20Finetuning%20and%20Adaptation/5%20Effective%20Training/4%20Pipeline%20Parallelism.md)  
  📄 [Tensor/Model Parallelism](1%20Theory/2%20Finetuning%20and%20Adaptation/5%20Effective%20Training/5%20Tensor_Model%20Parallelism.md)  
  📄 [Sharded Training and ZeRO](1%20Theory/2%20Finetuning%20and%20Adaptation/5%20Effective%20Training/6%20Sharded%20Training%20and%20ZeRO.md)  

- **Instruments**  
  📄 [PEFT Cheatsheet](1%20Theory/2%20Finetuning%20and%20Adaptation/6%20Instruments/1%20PEFT%20Cheatsheet.md)  
  📄 [Accelerate Cheatsheet](1%20Theory/2%20Finetuning%20and%20Adaptation/6%20Instruments/2%20Accelerate%20Cheatsheet.md)  
  📄 [DeepSpeed Cheatsheet](1%20Theory/2%20Finetuning%20and%20Adaptation/6%20Instruments/3%20DeepSpeed%20Cheatsheet.md)  

- **Long context**  
  📄 [NTK scaling](1%20Theory/2%20Finetuning%20and%20Adaptation/7%20Long%20Context/1%20NTK%20scaling.md)  
  📄 [Flash Attention](1%20Theory/2%20Finetuning%20and%20Adaptation/7%20Long%20Context/2%20Flash%20Attention.md)

---

## III. Prompt Engineering и контроль генерации  

- **In-context learning**  
  📄 [Zero, One and Few Shot](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/1%20In-Context%20Learning/1%20Zero,%20One%20and%20Few%20Shot.md)  
  📄 [Chain of Thought](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/1%20In-Context%20Learning/2%20Chain%20of%20Thoughts.md)  
  📄 [SelfAsk](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/1%20In-Context%20Learning/3%20SelfAsk.md)  
  📄 [ReAct](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/1%20In-Context%20Learning/4%20ReAct.md)

- **Decoding**  
  📄 [Decoding Parameters](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/2%20Decoding/1%20Decoding%20Parameters.md)  
  📄 [Deterministic Decoding](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/2%20Decoding/2%20Deterministic%20Decoding.md)  
  📄 [Stochastic and Hybrid Decoding](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/2%20Decoding/3%20Stochastic%20and%20Hybrid%20Decoding.md)

- **Reasoning**  
  📄 [Tree of Thoughts](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/3%20Reasoning/1%20Tree%20of%20Thoughts.md)  
  📄 [Multi-hop Reasoning](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/3%20Reasoning/2%20Multi-hop%20Reasoning.md)

- **Prompt evaluation**  
  📄 [Prompt Evaluation](1%20Theory/3%20Prompt%20Engineering%20and%20Generation%20Control/4%20Prompt%20Evaluation/1%20Prompt%20Evaluation.md)

---

## IV. Embeddings и векторные представления  

- **Text Vector Representations**  
  📄 [Text Vector Representations](1%20Theory/4%20Embeddings/1%20Text%20Vector%20Representations/1%20Dense%20and%20Sparse%20Embeddings.md)  
  📄 [Contextual Embeddings](1%20Theory/4%20Embeddings/1%20Text%20Vector%20Representations/2%20Contextual%20Embeddings.md)  
  📄 [Sentence Embeddings](1%20Theory/4%20Embeddings/1%20Text%20Vector%20Representations/3%20Sentence%20Embeddings.md)

- **Domain Adaptation**  
  📄 [Contrastive Learning](1%20Theory/4%20Embeddings/2%20Domain%20Adaptation/1%20Contrastive%20Learning.md)  
  📄 [Triplet Loss](1%20Theory/4%20Embeddings/2%20Domain%20Adaptation/2%20Triplet%20Loss.md)  
  📄 [Info-NCE](1%20Theory/4%20Embeddings/2%20Domain%20Adaptation/3%20InfoNCE.md)  
  📄 [Supervised Contrastive Loss](1%20Theory/4%20Embeddings/2%20Domain%20Adaptation/4%20Supervised%20Contrastive%20Loss.md)  
  📄 [Negatives Mining](1%20Theory/4%20Embeddings/2%20Domain%20Adaptation/5%20Negatives%20Mining.md)  
  📄 [Hard and Soft Negatives](1%20Theory/4%20Embeddings/2%20Domain%20Adaptation/6%20Hard%20and%20Soft%20Negatives.md)  
  📄 [MoCo and Memory Bank](1%20Theory/4%20Embeddings/2%20Domain%20Adaptation/7%20MoCo%20and%20Memory%20Bank.md)  

- **Text Clustering**  
  📄 [Text Clustering](1%20Theory/4%20Embeddings/3%20Text%20Clustering/1%20Text%20Clustering.md)  
  📄 [UMAP and DAPT/TAPT](1%20Theory/4%20Embeddings/3%20Text%20Clustering/2%20UMAP%20and%20DAPT%20TAPT.md)

- **Embedding Models**  
  📄 [GTE](1%20Theory/4%20Embeddings/4%20Embedding%20Models/1%20GTE.md)  
  📄 [BGE](1%20Theory/4%20Embeddings/4%20Embedding%20Models/2%20BGE.md)  
  📄 [E5](1%20Theory/4%20Embeddings/4%20Embedding%20Models/3%20E5.md)  
  📄 [MiniLM](1%20Theory/4%20Embeddings/4%20Embedding%20Models/4%20MiniLM.md)  
  📄 [Cohere Embed](1%20Theory/4%20Embeddings/4%20Embedding%20Models/5%20Cohere%20Embed.md)  
  📄 [Ada](1%20Theory/4%20Embeddings/4%20Embedding%20Models/6%20Ada.md)  
  📄 [SBERT](1%20Theory/4%20Embeddings/4%20Embedding%20Models/7%20SBERT.md)

---

## V. Анализ и отладка моделей  

- **Model Interpretation**  
  📄 [Model Interpretation](1%20Theory/5%20Debugging/1%20Model%20Interpretation/1%20Model%20Interpretation.md)

- **Diagnosis of Errors and Hallucinations**  
  📄 [Diagnosis of Errors and Hallucinations](1%20Theory/5%20Debugging/2%20Diagnosis%20of%20Errors%20and%20Hallucinations/1%20Diagnosis%20of%20Errors%20and%20Hallucinations.md)

- **Тестирование устойчивости (robustness)**  
  📄 [Robustness](1%20Theory/5%20Debugging/3%20Robustness/1%20Robustness.md)

---

## VI. LLM в системах поиска и рекомендаций

📄 [LLM Search and Recs Intro](1%20Theory/6%20LLM%20in%20Search%20and%20Rec/0%20LLM%20Search%20and%20Recs%20Intro.md)  
📄 [Retrieval Overview](1%20Theory/6%20LLM%20in%20Search%20and%20Rec/1%20Retrieval%20Overview.md)  
📄 [Retrieval Reranking](1%20Theory/6%20LLM%20in%20Search%20and%20Rec/2%20Retrieval%20Reranking.md)  
📄 [Answer Generation](1%20Theory/6%20LLM%20in%20Search%20and%20Rec/3%20Answer%20Generation.md)  
📄 [LangChain RAG Search Intro](1%20Theory/6%20LLM%20in%20Search%20and%20Rec/4%20LangChain%20RAG%20Search%20Intro.md)  