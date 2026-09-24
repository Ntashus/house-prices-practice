# 📘 Этап 0. Загрузка и первичный осмотр данных

> **Файл:** `stage_0_loading_and_overview.md`
> **Датасет:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)
> **Цель этапа:** понять, с чем работаем — размер, типы, пропуски, диапазоны значений, высококардинальные и константные признаки.
> **Инструменты:** `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 📑 Содержание этапа

1. [Цель и ожидаемый результат](#-цель-и-ожидаемый-результат)
2. [Импорты и настройки окружения](#-импорты-и-настройки-окружения)
3. [Загрузка данных](#-загрузка-данных)
4. [Задача 0.1. Первый взгляд на данные](#-задача-01-первый-взгляд-на-данные)
5. [Задача 0.2. Техническая информация](#-задача-02-техническая-информация)
6. [Задача 0.3. Сводная таблица по колонкам](#-задача-03-сводная-таблица-по-колонкам)
7. [Задача 0.4. Разделение признаков по типам](#-задача-04-разделение-признаков-по-типам)
8. [Задача 0.5. Поиск «мусорных» колонок](#-задача-05-поиск-мусорных-колонок)
9. [Вопросы для самопроверки](#-вопросы-для-самопроверки)
10. [Шаблон вывода этапа](#-шаблон-вывода-этапа)
11. [Чек-лист этапа](#-чек-лист-этапа)
12. [Шпаргалка по методам этапа](#-шпаргалка-по-методам-этапа)

---

## 🎯 Цель и ожидаемый результат

После прохождения этапа ты должна уметь:

- Загрузить датасет и убедиться, что размеры и структура совпадают с ожидаемыми.
- Быстро оценить «здоровье» данных: сколько пропусков, какие типы, какие диапазоны.
- Разделить признаки на числовые и категориальные, а также выделить «псевдочисловые» (которые на самом деле категориальные).
- Найти колонки-константы и высококардинальные признаки.
- Сформулировать первые гипотезы о том, что нужно будет делать на этапах 1–8.

**Ожидаемые размеры:**
- `train.csv` → `(1460, 81)`
- `test.csv` → `(1459, 80)` (без `SalePrice`)
- `data_description.txt` → текстовое описание всех 79 признаков

---

## 🛠 Импорты и настройки окружения

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки отображения pandas
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Настройки визуализации
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
```

> 💡 **Совет.** Если работаешь в VS Code / PyCharm, добавь `%matplotlib inline` для Jupyter-ноутбука.

---

## 📥 Загрузка данных

```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

Проверь, что размеры совпадают с ожидаемыми:

```python
print('train:', train.shape)
print('test :', test.shape)
```

**Ожидаемый вывод:**
```
train: (1460, 81)
test : (1459, 80)
```

Если размеры другие — проверь, что скачала именно файлы с Kaggle (вкладка **Data**), а не примеры.

---

## 👀 Задача 0.1. Первый взгляд на данные

### Что делаем
Смотрим на первые/последние/случайные строки, чтобы понять структуру таблицы.

```python
train.head()
train.tail()
train.sample(5, random_state=42)
```

### На что обратить внимание
- **Первая колонка** — `Id`, это идентификатор, не признак.
- **Последняя колонка train** — `SalePrice` (целевая).
- **В test нет `SalePrice`** — это нормально, мы его предсказываем.
- **Много колонок** — 79 признаков. Часть из них — категориальные, часть — числовые.

### 💡 Полезный приём
Быстро вывести «широкую» таблицу в компактном виде:
```python
train.head(3).T   # транспонирование — колонки становятся строками
```
Так удобнее читать, когда колонок много.

---

## 🔍 Задача 0.2. Техническая информация

### Что делаем
Смотрим типы, количество non-null значений, потребление памяти.

```python
train.info()
```

**Что искать в выводе:**
- `int64` / `float64` → числовые признаки
- `object` → категориальные (строки)
- `Non-Null Count` → сколько непустых значений
- `memory usage` → сколько RAM занимает

### Описательная статистика

```python
# Числовые признаки
train.describe()
```

**Что искать:**
- `min` / `max` — есть ли аномальные диапазоны?
- `mean` vs `50%` (медиана) — сильно ли расходятся? Если да — есть скос или выбросы.
- `std` — большая вариация?

```python
# Категориальные признаки
train.describe(include='object')
```

**Что искать:**
- `unique` — сколько уникальных категорий?
- `top` / `freq` — какая категория доминирует?
- `NaN` — есть ли пропуски в категориальных?

---

## 📊 Задача 0.3. Сводная таблица по колонкам

### Что делаем
Строим собственный обзор: тип, уникальные значения, пропуски, диапазон.

### Шаг 1. Пропуски

```python
missing = train.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
missing_pct = (missing / len(train) * 100).round(2)

missing_df = pd.DataFrame({
    'n_missing': missing,
    'pct_missing': missing_pct
})
missing_df
```

**Ожидаемый топ пропусков (в train):**

| Колонка | Пропусков | % |
|---|---|---|
| `PoolQC` | 1453 | 99.52 |
| `MiscFeature` | 1406 | 96.30 |
| `Alley` | 1369 | 93.77 |
| `Fence` | 1179 | 80.75 |
| `FireplaceQu` | 690 | 47.26 |
| `LotFrontage` | 259 | 17.74 |
| `GarageType` | 81 | 5.55 |
| ... | ... | ... |

> ⚠️ **Важно!** В этом датасете пропуски не всегда означают «данные потеряны». Для `PoolQC`, `Alley`, `Fence` — это значит «нет бассейна / нет аллеи / нет забора». Это будет подробно разбираться на **этапе 1**.

### Шаг 2. Сводная таблица по всем колонкам

```python
summary = pd.DataFrame({
    'dtype': train.dtypes,
    'n_unique': train.nunique(),
    'n_missing': train.isna().sum(),
    'pct_missing': (train.isna().sum() / len(train) * 100).round(2),
})

# Добавим min/max для числовых
num_cols_all = train.select_dtypes(include=[np.number]).columns
summary.loc[num_cols_all, 'min'] = train[num_cols_all].min()
summary.loc[num_cols_all, 'max'] = train[num_cols_all].max()

# Добавим top/freq для категориальных
for col in train.select_dtypes(include=['object']).columns:
    summary.loc[col, 'top'] = train[col].mode().iloc[0] if not train[col].mode().empty else None
    summary.loc[col, 'top_freq'] = train[col].value_counts().iloc[0] if not train[col].value_counts().empty else 0

summary.sort_values('pct_missing', ascending=False)
```

### Визуализация пропусков

```python
plt.figure(figsize=(14, 6))
sns.heatmap(train[missing.index].isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Карта пропусков (жёлтый = пропуск)')
plt.xlabel('Признаки')
plt.tight_layout()
plt.show()
```

> 💡 **Что искать:** паттерны. Если пропуски в `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` идут «одной полосой» — это значит, что дома без гаража. Это не случайные пропуски.

### Альтернатива: barplot пропусков

```python
plt.figure(figsize=(10, 8))
missing_pct_sorted = (train.isna().sum() / len(train) * 100).sort_values(ascending=False)
missing_pct_sorted = missing_pct_sorted[missing_pct_sorted > 0]
sns.barplot(x=missing_pct_sorted.values, y=missing_pct_sorted.index, palette='viridis')
plt.title('Доля пропусков по признакам (%)')
plt.xlabel('% пропусков')
plt.tight_layout()
plt.show()
```

---

## 🧩 Задача 0.4. Разделение признаков по типам

### Что делаем
Разделяем колонки на группы для дальнейшей работы.

```python
# Целевая переменная
target = 'SalePrice'

# ID — не признак
id_col = 'Id'

# Числовые
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in [target, id_col]]

# Категориальные
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

print(f'Числовых признаков: {len(num_cols)}')
print(f'Категориальных признаков: {len(cat_cols)}')
```

### ⚠️ Псевдочисловые признаки

Некоторые колонки имеют числовой тип, но по смыслу — **категориальные**. Их нужно знать, чтобы случайно не посчитать для них `mean()` или корреляцию.

```python
pseudo_categorical = ['MSSubClass', 'MoSold', 'YrSold']
# MSSubClass — код типа дома (20, 30, 60...), не число в арифметическом смысле
# MoSold — месяц продажи (1-12), это цикл
# YrSold — год продажи (2006-2010), это метка времени
```

Проверь это сама:
```python
for col in pseudo_categorical:
    print(f"{col}: уникальных={train[col].nunique()}, "
          f"min={train[col].min()}, max={train[col].max()}")
```

### 💡 Почему это важно
- `MSSubClass = 60` не в 3 раза больше, чем `MSSubClass = 20`.
- Корреляция Пирсона для `MSSubClass` с `SalePrice` бессмысленна.
- На этапе 5 их нужно будет обрабатывать как категории (или ординально, если есть порядок).

---

## 🗑 Задача 0.5. Поиск «мусорных» колонок

### Что делаем
Ищем колонки, которые не несут информации.

### Константы (нулевая дисперсия)

```python
constant_cols = [c for c in train.columns if train[c].nunique() == 1]
print('Константные колонки:', constant_cols)
```

**Ожидание:** в train таких, скорее всего, нет, но проверка быстрая. **Важно проверить также в test** — если колонка константна в train, но не в test, удалять её нельзя.

```python
# Проверка на объединённых данных
all_data = pd.concat([train.drop(columns=[target]), test], ignore_index=True)
constant_cols_all = [c for c in all_data.columns if all_data[c].nunique() == 1]
print('Константные в train+test:', constant_cols_all)
```

### Высококардинальные категориальные

```python
high_card = train[cat_cols].nunique().sort_values(ascending=False)
high_card.head(10)
```

**Ожидаемый топ:**
| Колонка | Уникальных |
|---|---|
| `Neighborhood` | 25 |
| `Exterior1st` | 15 |
| `Exterior2nd` | 16 |
| `SaleType` | 9 |
| `Condition1` | 9 |
| `HouseStyle` | 8 |

> 💡 **Что делать с высококардинальными:** на этапе 5 их можно либо оставить как есть (если данных хватает), либо сгруппировать редкие категории в `"Other"`. Решение принимаем по частоте: если категория встречается <1% раз, она может быть шумом.

### Колонки с почти нулевой вариацией

```python
# Категориальные: одна категория занимает >95%
low_var_cat = {}
for col in cat_cols:
    top_freq = train[col].value_counts(normalize=True).iloc[0]
    if top_freq > 0.95:
        low_var_cat[col] = round(top_freq * 100, 2)
print('Категориальные с доминирующей категорией >95%:', low_var_cat)
```

**Ожидание:**
| Колонка | Доля топ-категории, % |
|---|---|
| `Street` | 99.59 (Pave) |
| `Utilities` | 99.79 (AllPub) |
| `CentralAir` | 93.49 (Y) |
| `PoolQC` | ~99.5 (None, после обработки) |

Эти колонки — кандидаты на удаление (кроме тех, где редкое значение несёт важную информацию).

---

## ❓ Вопросы для самопроверки

Ответь себе (запиши в ноутбук):

1. **Сколько колонок имеют >50% пропусков?** Перечисли их.
2. **Какие категориальные признаки имеют больше 10 уникальных значений?** Это проблема или нет?
3. **Есть ли колонки-константы в train? А в test?**
4. **Какие числовые признаки на самом деле категориальные по смыслу?** Почему?
5. **Какие признаки имеют длинный правый хвост** (skew > 1)?
   ```python
   skewed = train[num_cols].skew().sort_values(ascending=False)
   print(skewed[skewed > 1])
   ```
6. **Какие признаки содержат отрицательные значения?** Логично ли это?
   ```python
   neg_cols = [c for c in num_cols if (train[c] < 0).any()]
   print(neg_cols)
   ```
7. **Сколько памяти занимает train?** Можно ли оптимизировать (например, `int64 → int32`)?

---

## 📝 Шаблон вывода этапа

Скопируй в ноутбук и заполни своими числами:

> ### Вывод по этапу 0
>
> Датасет `train.csv` содержит **1460 строк** и **81 колонку**, из которых:
> - **N числовых** признаков
> - **M категориальных** признаков
> - **1 целевая** (`SalePrice`)
> - **1 идентификатор** (`Id`)
>
> **Пропуски:** X колонок имеют пропуски. Топ-5 по доле пропусков: `PoolQC` (99.5%), `MiscFeature` (96.3%), `Alley` (93.8%), `Fence` (80.8%), `FireplaceQu` (47.3%). Важно: в этом датасете NA часто означает «нет объекта», а не «данные потеряны» — это будет разбираться на этапе 1.
>
> **Псевдочисловые признаки:** `MSSubClass`, `MoSold`, `YrSold` — числовые по типу, но категориальные по смыслу.
>
> **Константные / почти константные:** [...]
>
> **Высококардинальные:** `Neighborhood` (25), `Exterior1st` (15), `Exterior2nd` (16) — требуют внимания при feature engineering.
>
> **Следующий шаг:** этап 1 — разобраться с семантикой пропусков и обработать их.

---

## ✅ Чек-лист этапа

- [ ] Файлы `train.csv`, `test.csv`, `data_description.txt` скачаны
- [ ] Размеры `train` и `test` совпадают с ожидаемыми
- [ ] `train.info()` просмотрен, типы ясны
- [ ] `describe()` для числовых и категориальных просмотрен
- [ ] Сводная таблица по колонкам построена
- [ ] Карта пропусков построена
- [ ] Числовые и категориальные признаки разделены на `num_cols` / `cat_cols`
- [ ] Псевдочисловые признаки выделены в отдельный список
- [ ] Константы и высококардинальные найдены
- [ ] Ответы на 7 вопросов для самопроверки записаны
- [ ] Вывод этапа записан в ноутбук

---

## 📖 Шпаргалка по методам этапа

| Задача | Метод / атрибут | Пример |
|---|---|---|
| Загрузка CSV | `pd.read_csv` | `pd.read_csv('train.csv')` |
| Размер | `df.shape` | `(1460, 81)` |
| Первые строки | `df.head(n)` | `df.head()` |
| Последние строки | `df.tail(n)` | `df.tail()` |
| Случайные строки | `df.sample(n)` | `df.sample(5, random_state=42)` |
| Типы и non-null | `df.info()` | `df.info()` |
| Описательная статистика | `df.describe()` | `df.describe(include='object')` |
| Типы колонок | `df.dtypes` | `df.dtypes` |
| Выбор по типу | `df.select_dtypes` | `df.select_dtypes(include=[np.number])` |
| Уникальные | `df['col'].nunique()` | `train['Neighborhood'].nunique()` |
| Частоты | `df['col'].value_counts()` | `train['SaleType'].value_counts()` |
| Пропуски по колонкам | `df.isna().sum()` | `train.isna().sum()` |
| Доля пропусков | `df.isna().mean()` | `(train.isna().mean() * 100).round(2)` |
| Транспонирование | `df.T` | `train.head(3).T` |
| Скос | `df['col'].skew()` | `train['SalePrice'].skew()` |
| Куртозис | `df['col'].kurt()` | `train['SalePrice'].kurt()` |
| Объединение | `pd.concat` | `pd.concat([train, test])` |

---

**Конец этапа 0.** Переходи к [этапу 1 — работа с пропусками](#) после того, как заполнишь вывод и ответишь на вопросы.