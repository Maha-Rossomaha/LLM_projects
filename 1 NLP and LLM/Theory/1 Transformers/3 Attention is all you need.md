# Подробный конспект: “Attention Is All You Need” (Vaswani et al., 2017)

URL: https://arxiv.org/pdf/1706.03762

## 0  Аннотация  
Transformer — первая последовательностная архитектура, основанная **исключительно** на механизме внимания, без рекуррентных и сверточных компонентов. Модель состоит из 6 идентичных энкодер‑блоков и 6 декодер‑блоков, каждый из которых содержит Multi‑Head Attention и позиционно‑независимую Feed‑Forward MLP с резидуальными связями и LayerNorm. Введены **Scaled Dot‑Product Attention**, синусоидальное **Positional Encoding** и стратегия обучения с прогревом learning‑rate. На задаче перевода WMT‑14 Transformer достиг 28.4 BLEU (EN→DE) и 41.8 BLEU (EN→FR), превзойдя сверточные и RNN‑модели при 3–4‑кратном ускорении обучения.

---

## 1  Введение и мотивация  
До 2017 г. лучшие Seq2Seq‑модели полагались на RNN или CNN энкодеры c attention‑модулем. Transformer устраняет рекуррентные зависимости, что позволяет полностью распараллелить вычисления и значительно ускорить обучение и инференс.

---

## 2  Scaled Dot‑Product Attention  

$$
\mathrm{Attention}(Q,K,V)=
\operatorname{softmax}\!\Bigl(\tfrac{QK^{\top}}{\sqrt{d_k}}\Bigr)V,
$$

где $Q,K,V\in\mathbb{R}^{n\times d_k}$. Деление на $\sqrt{d_k}$ стабилизирует градиенты.

### 2.1  Маскирование для автогрессии  

$$
\mathrm{MaskedAttn}(Q,K,V)=
\operatorname{softmax}\!\Bigl(\tfrac{QK^{\top}}{\sqrt{d_k}}+M\Bigr)V,
$$

где $M_{ij}=-\infty$ при $j>i$ обеспечивает запрет «взгляда в будущее».

---

## 3  Multi‑Head Attention  

$$
\mathrm{head}_h=\mathrm{Attention}(QW_h^Q,KW_h^K,VW_h^V),$$
$$\mathrm{MHA}(Q,K,V)=\operatorname{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)W^O.
$$

В оригинале \(H=8\), \(d_k=64\), что позволяет модели извлекать информацию из разных подпространств.

---

## 4  Структура блочных модулей  

### 4.1  Энкодер‑блок  
`[Input] → MHA → FFN`, оба подслоя окружены Residual + LayerNorm.

### 4.2  Декодер‑блок  
`[Prev y] → Masked MHA → Cross MHA (Q от декодера, K,V от энкодера) → FFN`.

---

## 5  Positional Encoding  

$$
\mathrm{PE}_{(pos,2i)}=\sin\bigl(pos/10000^{2i/d_{\text{model}}}\bigr),\quad$$
$$\mathrm{PE}_{(pos,2i+1)}=\cos\bigl(pos/10000^{2i/d_{\text{model}}}\bigr).
$$

Синусоиды позволяют модели кодировать относительные расстояния и обобщать на более длинные последовательности.

---

## 6  Position‑Wise Feed‑Forward Network  

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)\,W_2+b_2,
$$

где $W_1\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}},\; d_{\text{ff}}=2048$.

---

## 7  Схема обучения  

| Гиперпараметр | Значение |
|--------------|----------|
| Оптимизатор  | Adam $(\beta_1=0.9,\ \beta_2=0.98)$ |
| Learning‑rate | $d_{\text{model}}^{-0.5}\!\cdot\!\min(\text{step}^{-0.5},\,\text{step}\cdot\text{warmup}^{-1.5}),\ \text{warmup}=4000$ |
| Dropout      | 0.1 |
| Label‑smoothing | $\varepsilon=0.1$ |

---

## 8  Результаты  

| Задача | BLEU |
|--------|------|
| WMT‑14 EN→DE | 28.4 |
| WMT‑14 EN→FR | 41.8 |

Transformer превосходит ConvS2S и GNMT, обучаясь втрое быстрее.

---

## 9  Комплексность  

- **Attention**: $O(n^{2}d)$ — квадратичная по длине, но полностью параллельна.  
- **RNN**: $O(nd^{2})$ — последовательная; Transformer эффективнее на GPU/TPU.

---

## 10  Влияние  
Transformer стал стандартом в NLP и вдохновил GPT, BERT, T5, ViT и др., подтвердив, что «attention is all you need».
