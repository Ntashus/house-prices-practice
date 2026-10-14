📘 Этап 1. Работа с пропусками — готовый артефакт

Файл: stage_1_missing/stage_1_missing.md — положи в папку stage_1_missing/ рядом с ноутбуком
Ноутбук: stage_1_missing.ipynb
Датасет: House Prices - Advanced Regression Techniques
Цель этапа: понять семантику NA в каждой колонке и обработать пропуски корректно.
📑 Содержание

Что делали на этапе
Задача 1.1. Изучение data_description.txt
Задача 1.2. Загрузка данных
Задача 1.3. Пропуски до обработки
Задача 1.4. Заполнение группы A
Задача 1.5. Заполнение группы B
Задача 1.6. Умная импутация LotFrontage
Задача 1.7. Заполнение Electrical
Задача 1.8. Заполнение пропусков в test
Задача 1.9. Сохранение очищенных данных
Вопросы для самопроверки
Финальный вывод по этапу 1
Методы, которые освоили
Артефакты на выходе
Что дальше
🎯 Что делали на этапе

Разбирались с пропусками в датасете. Ключевая идея — NA ≠ NA:

В одних колонках NA значит «нет объекта» (нет бассейна → нет PoolQC).
В других — «данные потеряны» (LotFrontage не измерили).
В третьих — «нет объекта у числовой колонки» (нет гаража → нет года постройки).
Правильная обработка зависит от семантики, а не от «поставь медиану везде».

Результат: 19 колонок в train + 15 в test — все пропуски заполнены, сохранены train_clean.csv и test_clean.csv.

📖 Задача 1.1. Изучение data_description.txt

Что делали

Открыли официальное описание датасета и прошлись по всем 19 колонкам с пропусками — что означает NA в каждой.

Как искали

bash
Копировать
Скачать
# Искали контекст вокруг названия колонки — 8 строк после
grep -A 8 "PoolQC" data/data_description.txt
grep -A 6 "Alley" data/data_description.txt
grep -A 8 "FireplaceQu" data/data_description.txt
# и так далее
Что нашли — три группы

Группа A — NA = «нет объекта» (категориальные) — 15 колонок

В описании прямо указано, что NA = отсутствие объекта:

Колонка	Что значит NA
PoolQC	No Pool
MiscFeature	No MiscFeature (None)
Alley	No alley access
Fence	No Fence
FireplaceQu	No Fireplace
GarageType	No Garage
GarageFinish	No Garage
GarageQual	No Garage
GarageCond	No Garage
BsmtQual	No Basement
BsmtCond	No Basement
BsmtExposure	No Basement
BsmtFinType1	No Basement
BsmtFinType2	No Basement
MasVnrType	No Veneer (None)
Что делать: заполнить "None" — сохранить факт «объекта нет» как категорию.

Группа B — NA = «нет объекта» (числовые) — 2 колонки

Колонка	Что значит NA
GarageYrBlt	Нет гаража → нет года постройки
MasVnrArea	Нет облицовки → нет площади
Что делать: заполнить 0.

Группа C — NA = «данные потеряны» — 2 колонки

Колонка	Пропусков	Что значит
LotFrontage	259 (17.7%)	Ширина фасада — реально не измерили
Electrical	1	Тип электрики — потерян
Что делать: умная импутация.

LotFrontage — медиана по Neighborhood (дома в одном районе похожи).
Electrical — мода (категориальная колонка).
Вывод

Прошли 15 колонок через grep -A N "<название>" data/data_description.txt. Подтвердили: NA в группе A — это не потеря данных, а факт «объекта нет».

🛠 Задача 1.2. Загрузка данных

Что делали

Загрузили оригинальные train.csv и test.csv (ещё с пропусками) и списки колонок из columns_info.json (сохранены на этапе 0).

Код

python
Копировать
Скачать
train = pd.read_csv("../data/train.csv")
test  = pd.read_csv("../data/test.csv")

with open("../columns_info.json") as f:
    cols_info = json.load(f)

target = cols_info["target"]
id_col = cols_info["id_col"]
pure_num_cols = cols_info["pure_num_cols"]
pseudo_num_cols = cols_info["pseudo_num_cols"]
cat_cols = cols_info["cat_cols"]
Что получили

text
Копировать
Скачать
train: (1460, 81)
test : (1459, 80)

Числовых: 33
Псевдочисловых: 3
Категориальных: 43
Вывод

Данные загружены, списки колонок подхвачены из этапа 0 — не нужно считать заново.

📊 Задача 1.3. Пропуски до обработки

Что делали

Посчитали все пропуски в train — это baseline для сравнения.

Код

python
Копировать
Скачать
missing_before = train.isna().sum().sort_values(ascending=False)
missing_before = missing_before[missing_before > 0]

print(f"Колонок с пропусками: {len(missing_before)}")
print(f"Всего пропусков: {train.isna().sum().sum()}")
print(missing_before)
Что увидели

text
Копировать
Скачать
Колонок с пропусками: 19
Всего пропусков: 7829

PoolQC          1453
MiscFeature     1406
Alley           1369
Fence           1179
MasVnrType       872
FireplaceQu      690
LotFrontage      259
GarageYrBlt       81
GarageCond        81
GarageType        81
GarageFinish      81
GarageQual        81
BsmtFinType2      38
BsmtExposure      38
BsmtQual          37
BsmtCond          37
BsmtFinType1      37
MasVnrArea         8
Electrical         1
Вывод

7829 пропусков в 19 колонках. Топ-5 по доле — PoolQC, MiscFeature, Alley, Fence, MasVnrType (все > 59%).

🗑 Задача 1.4. Заполнение группы A — «нет объекта» (категориальные)

Что делали

Для 15 категориальных колонок, где NA = «нет объекта», заменили NaN на строку "None".

Код

python
Копировать
Скачать
none_cat_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

print("Пропусков до:", train[none_cat_cols].isna().sum().sum())

for col in none_cat_cols:
    train[col] = train[col].fillna("None")
    test[col]  = test[col].fillna("None")

print("Пропусков после:", train[none_cat_cols].isna().sum().sum())
Результат

text
Копировать
Скачать
Пропусков до: 7480
Пропусков после: 0
Почему "None", а не медиана/мода

NA здесь = «объект отсутствует». Это не потеря, а информация.
Медиана или мода «затрёт» эту информацию: половина домов без бассейна → мода = «такой же, как у большинства» → теряем различие.
"None" — честная категория «нет объекта». Модель/анализ смогут её использовать.
Вывод

7480 пропусков заполнены строкой "None" в train и test. Осталось 349 в train, 342 в test — это группы B и C.

🔢 Задача 1.5. Заполнение группы B — «нет объекта» (числовые)

Что делали

Для двух числовых колонок (GarageYrBlt, MasVnrArea), где NA = «нет объекта», заполнили 0.

Код

python
Копировать
Скачать
none_num_cols = ["GarageYrBlt", "MasVnrArea"]

print("Пропусков до:", train[none_num_cols].isna().sum().sum())

for col in none_num_cols:
    train[col] = train[col].fillna(0)
    test[col]  = test[col].fillna(0)

print("Пропусков после:", train[none_num_cols].isna().sum().sum())
Результат

text
Копировать
Скачать
Пропусков до: 89
Пропусков после: 0
Почему 0, а не медиана

GarageYrBlt = 0 значит: «гаража нет → нет года». Логично.
GarageYrBlt = 1980 (медиана) значило бы: «гараж построен в 1980» — но гаража-то нет. Ложная информация.
Модель могла бы решить: «раз год есть, значит гараж есть» → ошибка.
Вывод

89 пропусков заполнены нулями. Осталось: LotFrontage (259) и Electrical (1).

🎯 Задача 1.6. Умная импутация LotFrontage

Что делали

Заполнили 259 пропусков в LotFrontage медианой по Neighborhood, а не глобальной медианой.

Почему не глобальная медиана

Дома в одном районе похожи по ширине фасада. Глобальная медиана «сглаживает» реальные различия между районами.

Проверка:

python
Копировать
Скачать
global_median = train["LotFrontage"].median()
print(f"Глобальная медиана: {global_median:.1f}")

neighborhood_medians = train.groupby("Neighborhood")["LotFrontage"].median()
print(f"Min медианы по районам: {neighborhood_medians.min():.1f}")
print(f"Max медианы по районам: {neighborhood_medians.max():.1f}")
Результат: глобальная ~69, по районам — от ~40 до ~90. Разброс большой → по районам честнее.

Код

python
Копировать
Скачать
print("Пропусков до:", train["LotFrontage"].isna().sum())

for df in [train, test]:
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"] \
        .transform(lambda x: x.fillna(x.median()))

print("Пропусков после:", train["LotFrontage"].isna().sum())
Разбор transform

groupby("Neighborhood") — группируем строки по районам.
["LotFrontage"] — берём колонку ширины фасада.
.transform(lambda x: x.fillna(x.median())) — внутри каждой группы считаем медиану и заполняем ею пропуски. Возвращает Series той же длины, что исходный DataFrame.
Результат

text
Копировать
Скачать
Пропусков до: 259
Пропусков после: 0
Вывод

259 пропусков заполнены медианой по районам. Это точнее, чем глобальная медиана.

⚡ Задача 1.7. Заполнение Electrical

Что делали

1 пропуск в категориальной колонке Electrical заполнили модой.

Код

python
Копировать
Скачать
print("До:", train["Electrical"].isna().sum())

electrical_mode = train["Electrical"].mode().iloc[0]
print(f"Мода Electrical: {electrical_mode}")

for df in [train, test]:
    df["Electrical"] = df["Electrical"].fillna(electrical_mode)

print("После:", train["Electrical"].isna().sum())
Разбор .mode().iloc[0]

.mode() возвращает Series — потому что мод может быть несколько (если две категории встречаются одинаково часто).
.iloc[0] берёт первую — чтобы получить одно значение для fillna.
В нашем случае мода одна: SBrkr (1334 из 1460 домов).
Результат

text
Копировать
Скачать
До: 1
Мода Electrical: SBrkr
После: 0
Вывод

1 пропуск заполнен модой SBrkr.

🎯 Задача 1.8. Заполнение пропусков в test

Что делали

В test осталось 22 пропуска в 15 колонках, которых не было в train. Заполнили их модой/медианой из train.

Почему пропуски в test другие

test — отдельная выборка. В ней могут быть дома с категориями, которых в train не было вовсе. Например, MSZoning = "C (all)" встречается только в test.

Список колонок с пропусками в test

Категориальные (7)	Числовые (8)
MSZoning	BsmtFullBath
Utilities	BsmtHalfBath
Functional	BsmtFinSF1
Exterior1st	BsmtFinSF2
Exterior2nd	BsmtUnfSF
SaleType	TotalBsmtSF
KitchenQual	GarageCars
—	GarageArea
Код

python
Копировать
Скачать
# Категориальные — мода из TRAIN
cat_cols_test = ["MSZoning", "Utilities", "Functional", "Exterior1st",
                 "Exterior2nd", "SaleType", "KitchenQual"]

for col in cat_cols_test:
    mode_val = train[col].mode().iloc[0]    # мода по train
    test[col] = test[col].fillna(mode_val)  # заполняем NaN этой модой

# Числовые — медиана из TRAIN
num_cols_test = ["BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1",
                 "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "GarageCars", "GarageArea"]

for col in num_cols_test:
    median_val = train[col].median()         # медиана по train
    test[col] = test[col].fillna(median_val) # заполняем NaN этой медианой
Результат

text
Копировать
Скачать
MSZoning: заполнено RL
Utilities: заполнено AllPub
Functional: заполнено Typ
Exterior1st: заполнено VinylSd
Exterior2nd: заполнено VinylSd
SaleType: заполнено WD
KitchenQual: заполнено TA
BsmtFullBath: заполнено 0.0
BsmtHalfBath: заполнено 0.0
BsmtFinSF1: заполнено 383.5
BsmtFinSF2: заполнено 0.0
BsmtUnfSF: заполнено 477.5
TotalBsmtSF: заполнено 991.5
GarageCars: заполнено 2.0
GarageArea: заполнено 480.0

Всего пропусков в test: 0
Почему мода/медиана из train

Это правило ML: нельзя «учиться» на test. Иначе — data leakage (утечка данных из теста в правила). Логика из train применяется к test.

Вывод

22 пропуска в test заполнены. Всего после всех шагов: 0 пропусков в train и test.

💾 Задача 1.9. Сохранение очищенных данных

Что делали

Сохранили очищенные DataFrames в новые файлы — оригиналы не трогали.

Код

python
Копировать
Скачать
train.to_csv("../data/train_clean.csv", index=False)
test.to_csv("../data/test_clean.csv", index=False)

print("Сохранено: train_clean.csv, test_clean.csv")
Почему новые имена

train.csv / test.csv — оригиналы (с пропусками). Нужны для воспроизводимости.
train_clean.csv / test_clean.csv — очищенные. Используются в этапах 2–8.
Результат

text
Копировать
Скачать
Сохранено: train_clean.csv, test_clean.csv
Проверка файлов на диске

python
Копировать
Скачать
import pandas as pd
train_check = pd.read_csv("../data/train_clean.csv")
test_check  = pd.read_csv("../data/test_clean.csv")

print("Файл train_clean.csv — пропусков:", train_check.isna().sum().sum())
print("Файл test_clean.csv  — пропусков:", test_check.isna().sum().sum())
Ожидаемо:

text
Копировать
Скачать
Файл train_clean.csv — пропусков: 0
Файл test_clean.csv  — пропусков: 0
Вывод

Оба файла чистые. Данные готовы к этапу 2.

❓ Вопросы для самопроверки

1. Сколько колонок имели пропуски до обработки?
19 в train, 15 дополнительных в test.

2. Что значит NA в колонке PoolQC?
«No Pool» — нет бассейна. Это не потеря данных, а факт.

3. Почему GarageYrBlt заполняется нулём, а не медианой?
NA = «нет гаража → нет года». Медиана дала бы ложное значение: «гараж построен в 1980», хотя гаража нет.

4. Почему LotFrontage заполняется медианой по Neighborhood, а не глобальной?
Дома в одном районе похожи по ширине фасада. Глобальная медиана «сглаживает» реальные различия между районами.

5. Почему в test мода берётся из train, а не из test?
Правило ML: нельзя использовать test для обучения. Иначе — data leakage.

6. Какие 3 группы пропусков бывают в этом датасете?

A: NA = «нет объекта» (категориальные) → "None".
B: NA = «нет объекта» (числовые) → 0.
C: NA = «данные потеряны» → умная импутация.
7. Сколько пропусков было в train и стало?
Было 7829. Стало 0.

8. Что значит train.isna().sum().sum()?
Первая .sum() — пропуски по каждой колонке (Series). Вторая .sum() — сумма по всем колонкам (одно число).

📋 Финальный вывод по этапу 1

Датасет: train.csv (1460 × 81), test.csv (1459 × 80).

Пропуски:

7829 пропусков в 19 колонках train — все заполнены.
7558 пропусков в test — все заполнены.
Обработка по группам:

Группа	Колонок	Метод	До	После
A — категориальные, «нет объекта»	15	fillna("None")	7480	0
B — числовые, «нет объекта»	2	fillna(0)	89	0
C — LotFrontage	1	Медиана по Neighborhood	259	0
C — Electrical	1	Мода (SBrkr)	1	0
Доп. — test-специфичные	15	Мода/медиана из train	22	0
Итого	19 + 15		7829 + 7558	0
Сохранено:

data/train_clean.csv
data/test_clean.csv
Главный урок: NA ≠ NA. Прежде чем заполнять — читай data_description.txt и определяй семантику пропуска.

Следующий шаг: этап 2 — одномерный EDA.

🛠 Методы, которые освоили

Задача	Метод
Пропуски по колонкам	df.isna().sum()
Всего пропусков	df.isna().sum().sum()
Заполнить константой	df["col"].fillna("None")
Заполнить нулём	df["col"].fillna(0)
Мода	df["col"].mode().iloc[0]
Медиана	df["col"].median()
Медиана по группам	df.groupby("cat")["col"].transform(lambda x: x.fillna(x.median()))
Список колонок с пропусками	df.isna().sum()[df.isna().sum() > 0].index
Сохранить CSV	df.to_csv("file.csv", index=False)
Проверка файлов	pd.read_csv("file.csv").isna().sum().sum()
Список файлов в папке	os.listdir("../data")
📦 Артефакты на выходе

1. Ноутбук stage_1_missing.ipynb — все шаги задокументированы, выводы в markdown-ячейках.

2. Файлы train_clean.csv, test_clean.csv — очищенные данные (0 пропусков). Используются на этапах 2–8.

3. Понимание семантики NA — знание, что NA в разных колонках значит разное. Это критично для следующего этапа.

