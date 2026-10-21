Этап 2. Одномерный анализ (EDA) — готовый артефакт

Файл: stage_2_univariate/stage_2_univariate.md — кладётся рядом с stage_2_univariate.ipynb
Датасет: House Prices - Advanced Regression Techniques
Цель этапа: понять распределение каждого признака по отдельности — форма, скос, диапазон, нули.
Источники: data/train_clean.csv (1460 × 81), data/test_clean.csv (1459 × 80) — очищенные данные с этапа 1.
Ноутбук: stage_2_univariate.ipynb
Артефакты на выходе: визуализации распределений, выводы о скошенности и почти-константах.
📑 Содержание

Что делали на этапе
Задача 2.1. Анализ SalePrice
Задача 2.2. Распределения числовых признаков
Задача 2.3. Распределения категориальных признаков
Задача 2.4. Почти-константы среди числовых
Вопросы для самопроверки
Финальный вывод по этапу 2
Методы, которые освоили
Типичные ошибки новичков
Что дальше
🎯 Что делали на этапе

На этапе 1 мы очистили данные — заполнили все пропуски. Теперь смотрим на форму каждого признака по отдельности.

Ключевые вопросы:

Как распределена целевая переменная SalePrice?
Какие числовые признаки сильно скошены вправо?
Где много нулей — и что это значит?
Какие категориальные признаки почти-константы?
Какие признаки дискретные (счётчики, оценки), а какие непрерывные?
Результат: понимание формы каждого признака — база для этапа 3 (связи с целевой) и этапа 5 (feature engineering).

🎯 Задача 2.1. Анализ SalePrice

🎯 Цель

Понять распределение целевой переменной — форму, скос, диапазон.

📚 Что практикуем

describe(), skew(), kurt(), histplot(), kdeplot(), np.log1p().

🛠 Код

python
Копировать
Скачать
# Описательные статистики SalePrice
print("SalePrice — описательные статистики:")
print(train[target].describe().round(2))
print()

# Skewness — асимметрия распределения
print(f"Skewness: {train[target].skew():.3f}")

# Kurtosis — «толщина» хвостов
print(f"Kurtosis: {train[target].kurt():.3f}")
📊 Результат

text
Копировать
Скачать
SalePrice — описательные статистики:
count     1460.00
mean    180921.20
std      79442.50
min      34900.00
25%     129975.00
50%     163000.00
75%     214000.00
max     755000.00

Skewness: 1.883
Kurtosis: 6.536
🖼️ Визуализация

python
Копировать
Скачать
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Левая — исходный SalePrice
sns.histplot(train[target], kde=True, ax=axes[0], bins=50, color="steelblue")
axes[0].set_title("SalePrice — исходное распределение")

# Правая — log1p(SalePrice)
sns.histplot(np.log1p(train[target]), kde=True, ax=axes[1], bins=50, color="seagreen")
axes[1].set_title("log1p(SalePrice) — логарифмированное")

plt.tight_layout()
plt.show()
🧠 Что такое skew (скос)

Skewness — мера асимметрии распределения.

Значение	Что значит
0	Симметрично (среднее = медиана)
> 0	Правый скос (длинный хвост справа, среднее > медианы)
< 0	Левый скос (длинный хвост слева, среднее < медианы)
У нас: mean (180921) > median (163000) → правый скос, skew ≈ 1.88 (сильный).

Причина: несколько дорогих домов (до $755k) тянут среднее вверх.

🧪 Логарифмирование

np.log1p(x) = log(1 + x) — безопасно для нулей.

Что происходит:

Большие значения «сжимаются» сильнее маленьких.
755000 → log(755001) ≈ 13.5
34900 → log(34901) ≈ 10.5
Skew после логарифма: 1.88 → 0.12. Распределение стало почти симметричным.

Зачем это нужно:

Для ML: линейные модели, KNN, нейросети работают лучше с симметричными данными.
Для визуализации: видно форму, а не «столбик слева».
Для статистики: средние и доверительные интервалы требуют нормальности.
Но на EDA мы работаем с исходной шкалой — чтобы видеть реальные цены. Логарифм — на этапе 8 (ML).

📝 Вывод по 2.1

SalePrice имеет правый скос (mean > median, skew ≈ 1.88).
Логарифмирование (np.log1p) нормализует распределение (skew ≈ 0.12).
Длинный правый хвост — реальные дорогие дома (до $755k), не ошибки.
Для ML-этапа логарифмирование может быть полезно. На EDA работаем с исходной шкалой.
📊 Задача 2.2. Распределения числовых признаков

🎯 Цель

Понять форму каждого из 33 числовых признаков — скос, диапазон, нули.

📚 Что практикуем

hist(), skew(), сравнение распределений, доля нулей.

🛠 Код — сетка гистограмм

python
Копировать
Скачать
# Сетка гистограмм по всем числовым признакам
n_cols = 4
n_rows = (len(pure_num_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
axes = axes.ravel()

for i, col in enumerate(pure_num_cols):
    axes[i].hist(train[col], bins=30, color="steelblue", edgecolor="white")
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(labelsize=8)

# Скрываем пустые subplots
for i in range(len(pure_num_cols), len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()
🧮 Формула n_rows — округление вверх

python
Копировать
Скачать
n_rows = (len(pure_num_cols) + n_cols - 1) // n_cols
Это классический трюк для округления вверх при целочисленном делении:

33 элемента, 4 колонки → (33 + 3) // 4 = 9 строк.
9 × 4 = 36 мест — 33 графика + 3 пустых.
Альтернатива: math.ceil(33 / 4) — тоже 9, но требует импорта.

🛠 Код — топ-10 по скосу

python
Копировать
Скачать
skew_vals = train[pure_num_cols].skew().sort_values(ascending=False)

print("=== Топ-10 с наибольшим скосом ===")
print(skew_vals.head(10).round(2))
print()
print("=== Топ-10 с наименьшим скосом ===")
print(skew_vals.tail(10).round(2))
📊 Результат — топ по скосу

text
Копировать
Скачать
=== Топ-10 с наибольшим скосом ===
MiscVal        24.45     ← экстремальный
PoolArea       14.24
3SsnPorch      11.07
LowQualFinSF    9.00
ScreenPorch     3.95
BsmtHalfBath    3.94
...
🛠 Код — доля нулей

python
Копировать
Скачать
zero_share = (train[pure_num_cols] == 0).mean().sort_values(ascending=False)
print("=== Топ-10 по доле нулей ===")
print(zero_share.head(10).round(3))
📊 Результат — доля нулей

text
Копировать
Скачать
=== Топ-10 по доле нулей ===
PoolArea        0.99
3SsnPorch       0.98
LowQualFinSF    0.98
MiscVal         0.96
BsmtHalfBath    0.94
ScreenPorch     0.92
BsmtFinSF2      0.89
EnclosedPorch   0.86
HalfBath        0.62
MasVnrArea      0.59
🧠 Что значит «полезны как есть/нет, а не сколько»

Проблема: PoolArea — 99% домов без бассейна. Признак не различает дома.

Что делать: создать бинарный признак HasPool = (PoolArea > 0):

1453 дома → 0 (нет бассейна).
7 домов → 1 (есть бассейн).
Почему это работает:

Дома с бассейном стоят в 2 раза дороже (mean $347k vs $179k).
Модель «видит» различие, а не «размазанные» числа.
Когда применять:

zero_share > 90% → всегда бинаризовать.
zero_share 50–90% → можно бинаризовать или оставить как число.
🧠 Что значит «дискретные признаки (1–10 / 0–8)»

Дискретные — принимают конечное число значений, обычно целые.

Примеры:

OverallQual (1–10) — оценка качества.
OverallCond (1–9) — оценка состояния.
BedroomAbvGr (0–8) — число спален.
FullBath (0–3) — число ванных.
GarageCars (0–4) — места в гараже.
Отличия от непрерывных:

Гистограмма выглядит «столбиками», а не плавной кривой.
Среднее имеет смысл (хоть и звучит странно: «2.87 спальни»).
Значения упорядочены → корреляция Пирсона работает.
Не надо one-hot, не надо логарифмировать.
Что с ними делать:

Оставить как числа.
Boxplot на этапе 3 — увидишь связь с SalePrice.
При необходимости — объединить редкие значения.
📝 Вывод по 2.2

Большинство числовых признаков имеют правый скос — длинный хвост в сторону больших значений.
Топ по скосу: MiscVal, PoolArea, 3SsnPorch, LowQualFinSF, ScreenPorch (skew > 3).
5 колонок с долей нулей > 90%: PoolArea, 3SsnPorch, LowQualFinSF, MiscVal, ScreenPorch.
Эти признаки полезны как «есть / нет» (бассейн, доп. постройка), а не «сколько».
OverallQual, OverallCond, BedroomAbvGr — дискретные (1–10 / 0–8), распределения «столбиками».
Проверила: дома с бассейном стоят в 2 раза дороже ($347k vs $179k), хоть их и 7 из 1460. Признак полезен, несмотря на редкость.
🏷️ Задача 2.3. Распределения категориальных признаков

🎯 Цель

Понять распределение каждого из 43 категориальных признаков — сколько категорий, какая доминирует.

📚 Что практикуем

nunique(), mode(), value_counts(), pd.DataFrame({...}).

🛠 Код — сводка по категориальным

python
Копировать
Скачать
cat_summary = pd.DataFrame({
    "n_unique": train[cat_cols].nunique(),      # сколько уникальных
    "top": [train[c].mode().iloc[0] for c in cat_cols],       # самая частая
    "top_freq": [train[c].value_counts().iloc[0] for c in cat_cols],  # её частота
})

cat_summary["top_share"] = (cat_summary["top_freq"] / len(train)).round(3)
cat_summary = cat_summary.sort_values("top_share", ascending=False)
cat_summary
📊 Что видим — топ по top_share

Колонка	n_unique	top	top_share
Utilities	2	AllPub	1.00
Street	2	Pave	1.00
Condition2	8	Norm	0.99
RoofMatl	8	CompShg	0.98
Heating	6	GasA	0.98
...			
Neighborhood	25	NAmes	0.15
🛠 Код — почти-константы

python
Копировать
Скачать
almost_const = cat_summary[cat_summary["top_share"] > 0.95]
print(f"Почти-констант (top_share > 0.95): {len(almost_const)}")
print(almost_const[["n_unique", "top", "top_share"]])
📊 Результат — 8 почти-констант

text
Копировать
Скачать
Почти-констант (top_share > 0.95): 8

             n_unique       top  top_share
Utilities           2    AllPub       1.00
Street              2      Pave       1.00
Condition2          8      Norm       0.99
RoofMatl            8   CompShg       0.98
Heating             6      GasA       0.98
GarageCond          5        TA       0.96
GarageQual          5        TA       0.95
LandSlope           3       Gtl       0.95
🧠 Доминирующие категории (78–87%)

Три колонки имеют одну сильно доминирующую категорию, но не почти-константу:

Колонка	Категория	Домов	Доля
MSZoning	RL	1151	78.8%
BldgType	1Fam	1220	83.6%
SaleType	WD	1267	86.8%
Что значит на практике:

Большинство домов — обычные жилые дома (RL), односемейные (1Fam), стандартные сделки (WD).
Редкие категории (13–21%) — важны для анализа (например, New новостройки дороже).
На этапе 5: не one-hot целиком, а объединить редкие или target encoding.
🖼️ Визуализация — barplot по 6 колонкам

python
Копировать
Скачать
show_cols = ["Neighborhood", "MSZoning", "BldgType", "HouseStyle",
             "SaleType", "SaleCondition"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, col in enumerate(show_cols):
    counts = train[col].value_counts().head(10)   # топ-10 категорий
    counts.plot(kind="bar", ax=axes[i], color="steelblue")
    axes[i].set_title(f"{col} (топ-10 категорий)")
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
Почему head(10): у Neighborhood 25 категорий — 25 столбиков нечитаемо. Топ-10 — компромисс.

🧠 Что такое высокая кардинальность

n_unique > 10 → высокая кардинальность. Требует особой обработки:

One-hot даст 25 колонок → нужны target encoding или объединение редких.
В нашем датасете:

Neighborhood — 25 (самая высокая).
Exterior2nd — 16.
Exterior1st — 15.
📝 Вывод по 2.3

8 категориальных колонок — почти-константы (top_share > 0.95):
Utilities, Street, Condition2, RoofMatl, Heating, GarageCond, GarageQual, LandSlope.
3 колонки — доминирующая категория (78–87%): MSZoning (RL), BldgType (1Fam), SaleType (WD).
Neighborhood — самая информативная: 25 категорий, топ занимает только 15%.
Высококардинальные (n_unique > 10): Neighborhood, Exterior2nd, Exterior1st.
🔢 Задача 2.4. Почти-константы среди числовых

🎯 Цель

Проверить: есть ли числовые признаки с очень низкой вариацией.

📚 Что практикуем

range, cv (коэффициент вариации), zero_share.

🛠 Код — сводка по числовым

python
Копировать
Скачать
num_summary = pd.DataFrame({
    "min": train[pure_num_cols].min(),
    "max": train[pure_num_cols].max(),
    "mean": train[pure_num_cols].mean(),
    "std": train[pure_num_cols].std(),
})
num_summary["range"] = num_summary["max"] - num_summary["min"]
num_summary["cv"] = (num_summary["std"] / num_summary["mean"]).round(3)
num_summary["zero_share"] = (train[pure_num_cols] == 0).mean().round(3)

num_summary.sort_values("cv").head(10)
📊 Результат

text
Копировать
Скачать
               min      max    mean     std    range    cv  zero_share
YearRemodAdd 1950.00 2010.00 1984.87   20.65    60.00  0.01  0.00
YearBuilt    1872.00 2010.00 1971.27   30.20   138.00  0.01  0.00
OverallCond     1.00    9.00    5.58    1.11     8.00  0.20  0.00
KitchenAbvGr    0.00    3.00    1.05    0.22     3.00  0.21  0.00
OverallQual     1.00   10.00    6.10    1.38     9.00  0.23  0.00
...
🧠 Что значит CV

CV = std / mean — относительный разброс.

CV	Что значит
~0	Признак почти константа
0.1–1	Нормальный разброс
> 1	Большой разброс
Особенность: CV плохо работает для счётных с нулями (обсуждали в этапе 0).

📊 Настоящих почти-констант нет

YearRemodAdd (CV 0.01), YearBuilt (CV 0.01) — не мусор. Это годы (1900–2010), у них маленький относительный разброс. Но абсолютный разброс — 60–138 лет.

🛠 Код — экстремальная доля нулей

python
Копировать
Скачать
high_zero = num_summary[num_summary["zero_share"] > 0.9].sort_values("zero_share", ascending=False)
print("Колонок с zero_share > 90%:", len(high_zero))
print(high_zero[["zero_share", "min", "max"]])
Ожидаемо: 5 колонок — PoolArea, 3SsnPorch, LowQualFinSF, MiscVal, ScreenPorch.

📝 Вывод по 2.4

Настоящих почти-констант среди числовых нет.
YearRemodAdd, YearBuilt, GarageYrBlt — CV ~ 0.01, но это годы, не мусор.
OverallQual, OverallCond — дискретные (1–10 / 1–9), полезные.
5 колонок с zero_share > 90% — кандидаты на бинаризацию на этапе 5.
❓ Вопросы для самопроверки

1. Что такое skew и как его интерпретировать?
Мера асимметрии распределения. 0 — симметрично, > 0 — правый скос, < 0 — левый скос. У SalePrice ≈ 1.88 — сильный правый.

2. Почему SalePrice имеет правый скос?
Есть несколько очень дорогих домов (до $755k), которые тянут среднее вверх. Большинство — 100–250k.

3. Зачем логарифмировать SalePrice?
Для ML: линейные модели требуют нормальности. Log1p нормализует: skew 1.88 → 0.12.

4. Сколько почти-констант среди категориальных?
8: Utilities, Street, Condition2, RoofMatl, Heating, GarageCond, GarageQual, LandSlope. У всех top_share > 0.95.

5. Сколько почти-констант среди числовых?
0. Все числовые признаки различают дома. Годы (YearBuilt, YearRemodAdd) имеют низкий CV, но не являются константами.

6. Что значит «полезны как есть/нет, а не сколько»?
Для признаков с zero_share > 90% (PoolArea, 3SsnPorch) бинаризация (> 0 → 1/0) информативнее, чем числа. Модель видит «есть/нет» чётче.

7. Какие колонки имеют доминирующую категорию (но не константу)?
MSZoning (RL, 78.8%), BldgType (1Fam, 83.6%), SaleType (WD, 86.8%). Редкие категории (13–21%) важны для анализа.

8. Что такое дискретные признаки?
Признаки с конечным числом целых значений: OverallQual (1–10), BedroomAbvGr (0–8). Гистограмма — «столбики». Корреляция работает, one-hot не нужен.

9. Почему Neighborhood — самая информативная категориальная?
25 категорий (максимум в датасете), топ занимает только 15%. Разнообразие → различает дома.

10. Что делать с почти-константами (8 категориальными)?
Кандидаты на удаление. Но сначала проверить глазами: например, CentralAir (93.5%) оставить — кондиционер важен.

📋 Финальный вывод по этапу 2

Датасет: train_clean.csv (1460 × 81), test_clean.csv (1459 × 80) — 0 пропусков.

SalePrice

Правый скос, skew ≈ 1.88.
log1p нормализует (skew ≈ 0.12).
Диапазон: 34900 – 755000.
Медиана $163k, среднее $180k — среднее завышено выбросами.
Числовые признаки (33)

Большинство скошены вправо — длинный правый хвост.
Топ по скосу: MiscVal (24.45), PoolArea (14.24), 3SsnPorch (11.07).
5 колонок с zero_share > 90%: PoolArea, 3SsnPorch, LowQualFinSF, MiscVal, ScreenPorch. Полезны как «есть / нет».
Настоящих почти-констант нет (CV > 0.01 у большинства; низкий CV у годов — не мусор).
OverallQual, OverallCond, BedroomAbvGr — дискретные (1–10 / 0–8).
Категориальные признаки (43)

8 почти-констант (top_share > 0.95):
Utilities, Street, Condition2, RoofMatl, Heating, GarageCond, GarageQual, LandSlope.
3 доминирующие категории (78–87%):
MSZoning (RL), BldgType (1Fam), SaleType (WD).
Neighborhood — самая информативная: 25 категорий, топ занимает только 15%.
Высококардинальные (n_unique > 10): Neighborhood (25), Exterior2nd (16), Exterior1st (15).
Ключевые наблюдения

Skew и логарифм — важные инструменты. Skew → понять природу, log1p → подготовить к ML.
zero_share > 90% — сигнал для бинаризации, а не удаления.
Почти-константы — кандидаты на удаление, но проверять глазами.
Доминирование ≠ неинформативность — редкие категории важны.
Следующий шаг: этап 3 — двумерный EDA (связи с целевой).

🛠 Методы, которые освоили

Задача	Метод
Описательные статистики	df["col"].describe()
Skew / kurtosis	df["col"].skew(), df["col"].kurt()
Логарифмирование	np.log1p(df["col"])
Гистограмма + KDE	sns.histplot(..., kde=True)
Сетка графиков	plt.subplots(n_rows, n_cols), .ravel()
Округление вверх (деление)	(a + b - 1) // b
Скрыть subplot	axes[i].axis("off")
Skew всех колонок	df[cols].skew().sort_values()
Доля нулей	(df[cols] == 0).mean()
Уникальные	df["col"].nunique()
Мода	df["col"].mode().iloc[0]
Частота топ-категории	df["col"].value_counts().iloc[0]
Сводная таблица	pd.DataFrame({...})
Barplot	df["col"].value_counts().head(10).plot(kind="bar")
Коэффициент вариации	df["col"].std() / df["col"].mean()
Что запомнить

Skew — про форму. Правый скос → mean > median → выбросы справа.
Log1p — для ML, не для EDA. На EDA работай с исходной шкалой — реальные цены.
zero_share > 90% → бинаризация. Признак полезен как «есть/нет».
Дискретные (1–10) — оставляй как числа. Не one-hot, не логарифм.
Почти-константы (top_share > 0.95) — кандидаты на удаление.
Доминирующие (78–87%) — оставить. Редкие категории важны.
n_unique > 10 — высокая кардинальность. Нужны target encoding / объединение.
head(10) для визуализации. Полные данные — в value_counts().
