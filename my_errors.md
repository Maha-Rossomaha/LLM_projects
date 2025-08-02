1. HybridNorm  
2. P-tuning
3. post-ln - плохо масштабируется вглубину, требует warm-up. Pre-LN улучшает стабильность градиентов, часто не требует warm-up scheduler
4. LayerNorm - нормализует вектор по признакам, 
5. Multi‑head attention: независимо проецируем Q, K, V для каждого head, считаем attention и конкатенируем, затем дополнительный линейный выход. Это позволяет моделировать разные типы зависимостей одновременно
6. Вы не указали masked self-attention в каждом слое декодера: в архитектуре Each decoder layer consists of three sublayers: the causally masked self-attention, the encoder-decoder attention, and the feed-forward network. Вы описали лишь cross-attention и FFN, пропустив первый sub-layer — masked self-attention.
7. Cross-attention НЕ использует causal (нижнетреугольную) маску, только padding‑mask для encoder-length — никакого “masking на q” нет: decoder query может нелегально смотреть только на pad-эффекты, но не маскирует будущем 
8. BatchNorm в Transformer не применяется. Скорее — исключительно LayerNorm или RMSNorm, и BatchNorm даже в версиях не рассматривается. Ваше объяснение некорректно: batch-norm здесь вообще не используется 
9. RMSNorm не содержит “вычитание среднего”, только деление на √(mean(x²)+ε) и масштабирование принадлежит learnable γ. Вы смешали с centering, но его нет в оригинальной RMSNorm 
10. Конкатенация multi-head attention — это операция после параллельных projection Q/K/V, затем concat и final linear layer. Вы утверждаете, что всё это — “одно матричное умножение”, но компьютационно используется именно множество голов и дополнительная O-преобразование — всё же визуально кажется одно, но логически раздельно. Конкатенация необходима для вывода размерности и даёт expressivity именно за счёт multiple heads в каждой attention-head сети