# 🏁 Yandex Final Project 

## 📚 О проекте

Этот репозиторий — финальный проект трёхмесячной специализации **"Машинное обучение"** в **Яндекс Лицее** (весна 2025).  
Задача — классификация изображений на 16 классов активностей. Основная метрика: **macro F1-score**.

🔗 Соревнование на Kaggle: [Yandex ML Intensive Spring 2025](https://www.kaggle.com/competitions/ml-intensive-yandex-academy-spring-2025/overview)  
🥇 Наше решение заняло **1 место** на приватном лидерборде.

## 🧵 TL;DR (Суть решения в двух словах)

> Шестимодельный взвешенный **soft-ансамбль** (ConvNeXt-вариации, EfficientNet-like, TinyViT'ы).  
> На логитах моделей дополнительно обучены градиентные бустинги: CatBoost и LightGBM.  
> Финальный классификатор — взвешенное объединение предсказаний:  
> **0.4 × CatBoost + 0.3 × LightGBM + 0.3 × Soft Voting ансамбль**.  
> **Score на приватной таблице Kaggle:** [0.77267](https://www.kaggle.com/competitions/ml-intensive-yandex-academy-spring-2025/leaderboard)


## 🧩 Архитектура решения

1. **Модульное обучение 6 моделей**  
   Каждая модель обучалась как отдельный эксперимент с индивидуальными гиперпараметрами.  
   👉 [Final Models](#final-models)

2. **Генерация логитов и ансамблирование**  
   Сохранили логиты на валидации, для soft-voting ансамбля подобрали веса через Optuna.  
   Там же применили Test-Time Augmentation (TTA).  
   👉 [Ensembling](#ensembling)

4. **Финальный мета-классификатор**  
   На логитах обучили CatBoost и LightGBM. Сделали финальный предикт.  
   👉 [Staking](#staking)
   
<a name="final-models"></a>
## 🧠 Наши модели

В финальный ансамбль вошли 6 **обученных с нуля** моделей.

В решении **не использовались предобученные модели** ([teacher](notebooks/060_teacher.ipynb)), и **архитектуры "из коробки"** ([tinyvit_test](notebooks/05_vit_from_box(test).ipynb)). 

📊 **F1-scores отобранных моделей на валидации (в хронологическом порядке):** (с учетом TTA, реальные ниже)

- [`miniconvnext`](#miniconvnext): **0.6969**
- [`tiny_vit_no_ssl`](#tiny_vit_no_ssl): **0.7207**
- [`tiny_vit_ssl`](#tiny_vit_ssl): **0.7252**
- [`enet_v2_m`](#enet_v2_m): **0.7147**
- [`convnextv2_base`](#convnextv2_base): **0.6958**
- [`convnextv2_small`](#convnextv2_small): **0.7188**

### 📊 Общее по моделям

**🔗 Ссылка на веса моделей → [Yandex Disk](https://disk.yandex.ru/d/DfZp4Qd3suAzPQ)**

Для обучения всех шести моделей использовался единый набор аугментаций:

- 📐 **RandomResizedCrop** — с варьированием масштаба от 50% до 100% и соотношения сторон от 0.75 до 1.33
- 🔄 **RandomHorizontalFlip** — применяется с вероятностью 50%
- 🎛 **RandAugment** — 2 случайные операции с интенсивностью 9 (из 14 возможных трансформаций)
- 🎨 **ColorJitter** — изменяются яркость, контрастность, насыщенность и оттенок
- 📉 **Normalize** — по каналам с рассчитанными на тренировочном датасете значениями `mean` и `std`
- 🚫 **RandomErasing** — стирание случайного прямоугольника в изображении с вероятностью 25% (у одной модели вероятность была 50%)
- 🧬 **MixUp / CutMix** — применялись на уровне батчей во всех моделях;  
  часть реализаций — собственные, часть — из библиотеки `timm`  

> 🧪 Были проведены эксперименты с более агрессивными аугментациями  
(см. [`src/augmentations.py`](src/augmentations.py)) при обучении EfficientNetV2 (см. [`091_strong_enet_v2_m.ipynb`](notebooks/091_strong_enetv2_m.ipynb)),  
но они не дали значительного прироста метрик и сильно ухудшали совместимость с TTA,  
поэтому было принято решение остановиться на умеренно-сильных трансформациях.

<a name="miniconvnext"></a>
### 🧱 miniconvnext

**📁 Обучение:** [`04_miniconvnext_strong.ipynb`](notebooks/04_miniconvnext_strong.ipynb)  
**🧱 Реализация:** [`src/models/miniconvnext.py`](src/models/miniconvnext.py)  
**🧠 Архитектура:** компактная версия ConvNeXt (~1.6M параметров)  
**⏳ Всего эпох:** 350 (в несколько этапов с разными scheduler'ами)  
**💾 Веса:** `convnext_last.pth` — [скачать с Yandex Disk](https://disk.yandex.ru/d/DfZp4Qd3suAzPQ)  
**🧪 F1-score (валидация):** **0.6969**


**⚙️ Конфигурация обучения**

| Параметр      | Значение                                |
|---------------|-----------------------------------------|
| Оптимайзер    | `AdamW`, `lr=3e-5`, `weight_decay=1e-4` |
| Scheduler     | `OneCycleLR` / `CosineAnnealingLR`      |
| Эпох          | от 50 до 100 (в несколько этапов)       |
| Loss          | `CrossEntropyLoss` + `LabelSmoothing=0.1` |
| MixUp         | `alpha=1.0`, `p=0.2–0.5`             |
| CutMix        | `alpha=1.0`, `p=0.5–0.8`                 |

📝 Обучение производилось в несколько этапов с изменениями scheduler и параметров аугментаций.  
Подробности можно найти в [`ноутбуке`](notebooks/04_miniconvnext_strong.ipynb).

<a name="tiny_vit_no_ssl"></a>
### 🧱 tiny_vit_no_ssl

**📁 Обучение:** [`080_tinyvit_11m.ipynb`](notebooks/080_tinyvit_11m.ipynb)  
**🧱 Реализация:** [`src/models/tiny_vit.py`](src/models/tinyvit.py)  
**🧠 Архитектура:** базовая реализация `TinyViT-11M`  
**⏳ Всего эпох:** 275  
**💾 Веса:** `my_vit_4_best.pth` — [скачать с Yandex Disk](https://disk.yandex.ru/d/DfZp4Qd3suAzPQ)  
**🧪 F1-score (валидация):** **0.7207**

**⚙️ Конфигурация обучения**

| Параметр      | Значение                                |
|---------------|-----------------------------------------|
| Оптимайзер    | `AdamW`, `lr=3e-4`, `weight_decay=1e-4` |
| Scheduler     | `OneCycleLR` / `CosineAnnealingLR`      |
| Эпох          | от 50 до 75 (в несколько этапов)       |
| Loss          | `CrossEntropyLoss` + `LabelSmoothing=0.1` |
| MixUp         | `alpha=0.25-1.0`, `p=0.5`             |
| CutMix        | `alpha=0.25-1.0`, `p=0.5`                 |

📝 Обучение производилось в несколько этапов с изменениями scheduler и параметров аугментаций.  
Подробности можно найти в [`ноутбуке`](notebooks/080_tinyvit_11m.ipynb).

<a name="tiny_vit_ssl"></a>
### 🧱 tiny_vit_ssl

**📁 Обучение:** [`071_SSL.ipynb`](notebooks/071_SSL.ipynb)  
**🧱 Реализация:** [`src/models/tiny_vit.py`](src/models/tinyvit.py)  
**🧠 Архитектура:** `TinyViT-11M`, предварительно обученный через MAE (Masked Autoencoding)  
**🧪 SSL предобучение:** реализовано в ноутбуке с нуля (см. `MAEWrapper` внутри [`071_SSL.ipynb`](notebooks/071_SSL.ipynb))  
**🗂️ Использовался тот же датасет, что и для обучения** (через [`MAEDataset`](src/mae_dataset.py) — простая подача картинок)  
**⏳ Всего эпох:** 435  
**💾 Веса:** `ssl_vit_1_5_best.pth` — [скачать с Yandex Disk](https://disk.yandex.ru/d/DfZp4Qd3suAzPQ)  
**🧪 F1-score (валидация):** **0.7252**

**⚙️ Конфигурация предобучения (MAE)**

| Параметр         | Значение                                |
|------------------|------------------------------------------|
| Epochs           | 50                                       |
| Loss             | `MSELoss()`                              |
| Optimizer        | `AdamW(lr=3e-4, weight_decay=1e-3)`      |
| Scheduler        | `CosineAnnealingLR(T_max=50, eta_min=1e-6)` |

**⚙️ Конфигурация обучения**

| Параметр      | Значение                                |
|---------------|-----------------------------------------|
| Оптимайзер    | `AdamW`, `lr=3e-3 - 3e-4`, `weight_decay=1e-4` |
| Scheduler     | `OneCycleLR` / `CosineAnnealingLR`      |
| Эпох          | от 10 до 75 (в несколько этапов)       |
| Loss          | `CrossEntropyLoss` + `LabelSmoothing=0.1` |
| MixUp         | `alpha=0.25-1.0`, `p=0.5`             |
| CutMix        | `alpha=0.25-1.0`, `p=0.5`                 |

📝 Обучение производилось в несколько этапов с изменениями scheduler и параметров аугментаций.  
Подробности можно найти в [`ноутбуке`](notebooks/071_SSL.ipynb).

<a name="enet_v2_m"></a>
### 🧱 enet_v2_m

**📁 Обучение:** [`090_enetv2_m.ipynb`](notebooks/090_enetv2_m.ipynb)  
**🧱 Реализация:** [`src/models/enet.py`](src/models/enet.py)  
**🧠 Архитектура:** кастомная реализация EfficientNetV2-M (~50M параметров, против ~80M в оригинале)  
**⏳ Всего эпох:** 350  
**💾 Веса:** `enet_1_5_best.pth` — [скачать с Yandex Disk](https://disk.yandex.ru/d/DfZp4Qd3suAzPQ)  
**🧪 F1-score (валидация):** **0.7147**

**⚙️ Конфигурация обучения**

| Параметр      | Значение                                |
|---------------|-----------------------------------------|
| Оптимайзер    | `AdamW`, `lr=3e-3 - 3e-4`, `weight_decay=1e-4` |
| Scheduler     | `OneCycleLR` / `CosineAnnealingLR`      |
| Эпох          | от 50 до 75 (в несколько этапов)       |
| Loss          | `CrossEntropyLoss` + `LabelSmoothing=0.1` |
| MixUp         | `alpha=0.5-1.0`, `p=0.5`             |
| CutMix        | `alpha=0.5-1.0`, `p=0.5`                 |

📝 Обучение производилось в несколько этапов с изменениями scheduler и параметров аугментаций.  
Подробности можно найти в [`ноутбуке`](notebooks/090_enetv2_m.ipynb).

<a name="convnextv2_base"></a>
### 🧱 convnextv2_base

**📁 Обучение:** у него нет своего ноутбука, так как его пайплайн обучения ПОЛНОСТЬЮ СОВПАДАЕТ с ноутбуком convnextv2_small (но если нужно у меня есть все логи обучения в wandb)  
**🧱 Реализация:** [`src/models/convnextv2.py`](src/models/convnextv2.py)  
**🧠 Архитектура:** `ConvNeXtV2-Base`, реализована с нуля без предобучения и внешних данных  
**📦 Параметры:** ~87M (соответствует оригинальной версии от Meta AI)  
**⏳ Всего эпох:** 250  
**💾 Веса:** `convnextv2_base_3_best.pth` — [скачать с Yandex Disk](https://disk.yandex.ru/d/DfZp4Qd3suAzPQ)  
**🧪 F1-score (валидация):** **0.6958**

**⚙️ Конфигурация обучения**

| Параметр    | Значение                                                                 |
|-------------|--------------------------------------------------------------------------|
| Оптимайзер  | `AdamW`, `lr=3e-4`, `betas=(0.9, 0.999)`, `weight_decay=0.05`            |
| Scheduler   | `OneCycleLR` / `CosineAnnealingLR`                |
| Эпох        | 200                                                                       |
| Loss        | `CrossEntropyLoss` + `LabelSmoothing=0.1`                               |
| AMP         | `torch.amp.GradScaler()` (mixed precision)                              |
| MixUp       | `alpha=0.8`, `p=1.0`, `switch_prob=0.5`                                  |
| CutMix      | `alpha=1.0`, `p=1.0`, `switch_prob=0.5`                                  |

📝 Обучение производилось в несколько этапов с изменениями scheduler и параметров аугментаций.  
Подробности можно найти в [`ноутбуке`](notebooks/100_convnextv2small.ipynb).

<a name="convnextv2_small"></a>
### 🧱 convnextv2_small

**📁 Обучение:** [`100_convnextv2small.ipynb`](notebooks/100_convnextv2small.ipynb)  
**🧱 Реализация:** [`src/models/convnextv2.py`](src/models/convnextv2.py)  
**🧠 Архитектура:** `ConvNeXtV2-Small`, реализована с нуля без предобучения и внешних данных  
**📦 Параметры:** ~50M (соответствует оригинальной версии от Meta AI)  
**⏳ Всего эпох:** 200  
**💾 Веса:** `convnextv2_small_3_best.pth` — [скачать с Yandex Disk](https://disk.yandex.ru/d/DfZp4Qd3suAzPQ)  
**🧪 F1-score (валидация):** **0.7188**

**⚙️ Конфигурация обучения**

| Параметр    | Значение                                                                 |
|-------------|--------------------------------------------------------------------------|
| Оптимайзер  | `AdamW`, `lr=3e-4`, `betas=(0.9, 0.999)`, `weight_decay=0.05`            |
| Scheduler   | `OneCycleLR` / `CosineAnnealingLR`                |
| Эпох        | 200                                                                       |
| Loss        | `CrossEntropyLoss` + `LabelSmoothing=0.1`                               |
| AMP         | `torch.amp.GradScaler()` (mixed precision)                              |
| MixUp       | `alpha=0.8`, `p=1.0`, `switch_prob=0.5`                                  |
| CutMix      | `alpha=1.0`, `p=1.0`, `switch_prob=0.5`                                  |

📝 Обучение производилось в несколько этапов с изменениями scheduler и параметров аугментаций.  
Подробности можно найти в [`ноутбуке`](notebooks/100_convnextv2small.ipynb).

<a name="ensembling"></a>
## ⚖️ Ансамблирование

### 💾 Генерация логитов на Валидации

Для объединения моделей в ансамбль мы написали класс **WeightedEnsemble** ([`ensemple.py`](src/ensemble.py)). Этот класс позволял:

- 🧮 вычислять **взвешенные предсказания**;
- 🧾 генерировать **логиты**;
- 📊 рассчитывать **F1-score**;
- 📤 формировать **submission-файл**.

С помощью этого же класса были сохранены логиты, которые затем использовались:

- для оптимизации **весов soft-ансамбля**;
- для обучения **stacking-моделей** (CatBoost и LightGBM).

### 🔁 Использование Test-Time Augmentation (TTA)

> ⚡ **Одно из самых ценных открытий проекта!**  
> В ряде случаев **TTA повышало итоговый F1-score отдельных моделей на +0.03 и более** — это **ОЧЕНЬ много**.

TTA (Test-Time Augmentation) — это техника, при которой одно и то же изображение при инференсе проходит **несколько аугментаций**, а предсказания усредняются.

В нашем случае из одного изображения формировались **3 версии**:

1. 🖼 `Resize(256) → CenterCrop(224)`  
2. 🔄 То же, но с **горизонтальным отражением**  
3. ✂️ `RandomResizedCrop(224, scale 0.9–1.0)` + отражение

Все варианты **нормализовались** с `mean/std`, рассчитанными по тренировочному датасету.

### 🧮 Подбор весов soft-ансамбля через Optuna

- ⚙️ Веса подбирались с помощью библиотеки **Optuna**  
- 🎯 Цель — максимизация **macro F1-score** на валидации  
- ✅ Использовались как для **soft-voting**, так и для **мета-классификатора**

### 🔗 Ссылки

- 📁 Реализация ансамбля: [`ensemple.py`](src/ensemble.py)
- 📓 Подробности: [`110_final.ipynb`](notebooks/110_final.ipynb).

<a name="staking"></a>
## 🧠 Финальный мета-классификатор (Stacking)

После построения soft-voting ансамбля следующим логичным шагом стало применение **стекинга** — техники, при которой предсказания базовых моделей используются в качестве признаков для обучения нового (meta) классификатора.

---

### 🧩 Что использовалось в качестве признаков

- 🔢 **Логиты всех моделей**, сохранённые на валидации
- 🧠 **Мета-признаки**, извлечённые из логитов:
  - 📈 `confidence` — максимальная вероятность
  - 📊 `entropy` — мера неопределённости предсказания
  - ➖ `margin` — разница между двумя топовыми классами

🔗 Все признаки были объединены в единый `X` и использовались для обучения моделей.

---

### 🧪 Обучение стекинг-моделей

- 🚀 Модели-мета-классификаторы:
  - 🐱 **CatBoostClassifier**
  - 🌱 **LightGBMClassifier**

- ✅ Кросс-валидация: **Stratified K-Fold**, 5 фолдов  
- 📐 Целевая метрика: `macro F1-score`

---

### 🧬 Финальная формула классификатора

Финальное предсказание — **взвешенное объединение трёх источников**:

- `0.4 × CatBoost`
- `0.3 × LightGBM`
- `0.3 × Soft-voting Optuna ансамбль`

📦 Именно эта формула использовалась для генерации последнего `submission.csv`.

---

### 🔗 Полезные ссылки

- 📓 Подробности: [`110_final.ipynb`](notebooks/110_final.ipynb)

