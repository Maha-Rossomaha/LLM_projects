# Конспект главы 6 курса Hugging Face LLM (RU)

> Разделы: **6/2 — «Обучение нового токенизатора на основе старого»** и **6/8 — «Создание токенизатора, блок за блоком»**

---

## 1. Раздел 6/2 — Обучение нового токенизатора на основе старого

### 1.1  Зачем переучивать токенизатор
* **Другой язык / домен.** Если существующая LLM не поддерживает нужный язык или корпус резко отличается (медицинский, код и т. д.), — нужен «словарь» под новые данные.
* **Контроль над размером словаря.** Влияет на скорость, стоимость и объём памяти модели.
* **Сохранение архитектуры.** Переучиваем только словарь, не меняя алгоритм, специальных токенов и настроек.

### 1.2  Отличие от обучения модели
| Обучение токенизатора | Обучение модели |
| --- | --- |
| Детеминированная статистическая процедура (одинаковый результат при тех же данных). | Стохастический градиентный спуск; рандомизация, нужно фиксировать seed. |

### 1.3  Сбор и подготовка корпуса
```python
from datasets import load_dataset
raw_ds = load_dataset("code_search_net", "python")

# Генератор по 1000 функций, экономим RAM
def get_training_corpus():
    for i in range(0, len(raw_ds["train"]), 1000):
        yield raw_ds["train"][i : i + 1000]["whole_func_string"]
training_corpus = get_training_corpus()
```
*Использовали CodeSearchNet (1,6 GB Python‑кода) как пример специализированного корпуса.*

### 1.4  Переобучение словаря
```python
from transformers import AutoTokenizer
old_tok = AutoTokenizer.from_pretrained("gpt2")
new_tok = old_tok.train_new_from_iterator(training_corpus, vocab_size=52_000)
```
* Требуется **fast‑токенизатор** (библиотека *tokenizers* на Rust).
* Получаем тот же алгоритм GPT‑2 (byte‑level BPE), но со свежим словарём.

### 1.5  Проверка качества
```python
example = """def add_numbers(a, b):\n    return a + b"""
print(len(old_tok.tokenize(example)), len(new_tok.tokenize(example)))  # 36 → 27
```
* Новый словарь объединяет типичные для Python конструкции (`ĊĠĠĠ`, `Ġ"""`) и экономит токены.

### 1.6  Сохранение и публикация
```python
new_tok.save_pretrained("code-search-net-tokenizer")
# huggingface-cli login  ➜  new_tok.push_to_hub("<org>/code-search-net-tokenizer")
```

---

## 2. Раздел 6/8 — Создание токенизатора **с нуля**

### 2.1  Пять этапов конвейера
1. **Normalizer** — приведение текста (Unicode NFD/NFKD, lowercase, удаление акцентов…).
2. **Pre‑Tokenizer** — разбиение на словесные/байтовые фрагменты.
3. **Model** — алгоритм подслов: *WordPiece, BPE, Unigram…*
4. **Post‑Processor** — добавление спец‑токенов, масок, type‑id.
5. **Decoder** — обратное преобразование id → текст.

<p align="center"><em>Все блоки взаимозаменяемы в библиотеке <code>tokenizers</code>.</em></p>

### 2.2  Модули библиотеки
| Подмодуль | Содержит |
| --- | --- |
| `normalizers` | Примитивы и готовые нормализаторы (BertNormalizer, Lowercase…). |
| `pre_tokenizers` | Whitespace, ByteLevel, Metaspace, Punctuation… |
| `models` | `BPE`, `WordPiece`, `Unigram`, … |
| `trainers` | Один тренер на каждый алгоритм (`BpeTrainer`, `WordPieceTrainer`, …). |
| `post_processors` | `TemplateProcessing`, `ByteLevel`, … |
| `decoders` | `WordPiece`, `ByteLevel`, `Metaspace`, … |

### 2.3  Подготовка корпуса (пример WikiText‑2)
```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def get_training_corpus():
    for i in range(0, len(ds), 1000):
        yield ds[i : i + 1000]["text"]
```

### 2.4  **WordPiece** с нуля (BERT‑style)
```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers,
                         trainers, processors, decoders
# 1) модель
wp_tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
# 2) нормализация
wp_tok.normalizer = normalizers.Sequence([
    normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
])
# 3) пред‑токенизация
wp_tok.pre_tokenizer = pre_tokenizers.Whitespace()
# 4) обучение
trainer = trainers.WordPieceTrainer(
    vocab_size=25_000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)
wp_tok.train_from_iterator(get_training_corpus(), trainer=trainer)
# 5) пост‑процессор (CLS/SEP) и decoder
cls_id = wp_tok.token_to_id("[CLS]")
sep_id = wp_tok.token_to_id("[SEP]")
wp_tok.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[["[CLS]", cls_id], ["[SEP]", sep_id]]
)
wp_tok.decoder = decoders.WordPiece(prefix="##")
```

### 2.5  **Byte‑level BPE** (GPT‑2‑style)
```python
bpe_tok = Tokenizer(models.BPE())
bpe_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=25_000,
                             special_tokens=["<|endoftext|>"])
bpe_tok.train_from_iterator(get_training_corpus(), trainer=trainer)
bpe_tok.post_processor = processors.ByteLevel(trim_offsets=False)
bpe_tok.decoder = decoders.ByteLevel()
```

### 2.6  **Unigram** (XLNet‑style)
```python
uni_tok = Tokenizer(models.Unigram())
uni_tok.normalizer = normalizers.Sequence([
    normalizers.Replace("``", '"'), normalizers.Replace("''", '"'),
    normalizers.NFKD(), normalizers.StripAccents()
])
uni_tok.pre_tokenizer = pre_tokenizers.Metaspace()
trainer = trainers.UnigramTrainer(vocab_size=25_000,
        special_tokens=["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"],
        unk_token="<unk>")
uni_tok.train_from_iterator(get_training_corpus(), trainer=trainer)
cls_id = uni_tok.token_to_id("<cls>")
sep_id = uni_tok.token_to_id("<sep>")
uni_tok.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[["<sep>", sep_id], ["<cls>", cls_id]]
)
uni_tok.decoder = decoders.Metaspace()
```

### 2.7  Обёртка в Transformers
```python
from transformers import PreTrainedTokenizerFast
fast_tok = PreTrainedTokenizerFast(tokenizer_object=wp_tok,
                                   unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]",
                                   mask_token="[MASK]")
```
*Для GPT‑2 достаточно указать `<|endoftext|>` как bos/eos; для XLNet — `padding_side="left"`.*

### 2.8  Сохранение и публикация
```python
wp_tok.save("tokenizer.json")
fast_tok.save_pretrained("my-wp-tokenizer")
fast_tok.push_to_hub("<org>/my-wp-tokenizer")
```

---

## 3. Ключевые выводы
* **train_new_from_iterator()** — быстрый способ «перевырастить» словарь, сохранив всё остальное.
* **tokenizers** даёт полный контроль: можно менять любой блок конвейера и комбинировать их.
* Обязательно проверяйте: длину токенов, восстановление текста, покрытие доменных терминов.
* Публикация на Hub облегчает совместное использование и дальнейшее обучение LLM.

---

## 4. Полезные ссылки
* Раздел 6/2: https://huggingface.co/learn/llm-course/ru/chapter6/2
* Раздел 6/8: https://huggingface.co/learn/llm-course/ru/chapter6/8
* Библиотека `tokenizers`: https://github.com/huggingface/tokenizers

