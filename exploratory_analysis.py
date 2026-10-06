import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "SeoulBikeData.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.show()
    plt.close()


# ============================================================
# 1. Загрузка данных
# ============================================================

df = pd.read_csv(DATA_FILE, encoding="unicode_escape")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

# ============================================================
# 2. Первичная проверка качества данных
# ============================================================

print_section("ОБЩАЯ ИНФОРМАЦИЯ")
print("Размер датасета:", df.shape)
print("Количество строк:", len(df))
print("Количество столбцов:", len(df.columns))

print_section("СТОЛБЦЫ")
print(df.columns.tolist())

print_section("ПРОПУСКИ")
print(df.isnull().sum())

print_section("ДУБЛИКАТЫ")
print("Полных дубликатов:", df.duplicated().sum())
print("Дубликатов Date + Hour:", df.duplicated(subset=["Date", "Hour"]).sum())

print_section("ДАТЫ")
print("Первая дата:", df["Date"].min().strftime("%d/%m/%Y"))
print("Последняя дата:", df["Date"].max().strftime("%d/%m/%Y"))
print("Количество уникальных дат:", df["Date"].nunique())

print_section("ЧАСЫ")
print("Минимальный час:", df["Hour"].min())
print("Максимальный час:", df["Hour"].max())

print_section("СЕЗОНЫ")
print(df["Seasons"].value_counts())

print_section("ПРАЗДНИКИ")
print(df["Holiday"].value_counts())

print_section("РАБОЧИЕ ДНИ")
print(df["Functioning Day"].value_counts())

# ============================================================
# 3. Подготовка производных признаков
# ============================================================

df["Weekday"] = df["Date"].dt.day_name()
df["Day Type"] = df["Date"].dt.weekday.map(
    lambda x: "Weekend" if x >= 5 else "Weekday"
)
df["Rain"] = df["Rainfall(mm)"] > 0
df["Snow"] = df["Snowfall (cm)"] > 0

# ============================================================
# 4. Анализ по часам
# ============================================================

print_section("АНАЛИЗ ПО ЧАСАМ")

hourly = df.groupby("Hour")["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max"]
)
print(hourly)

average_by_hour = df.groupby("Hour")["Rented Bike Count"].mean()
max_hour = average_by_hour.idxmax()
min_hour = average_by_hour.idxmin()

print("\nСамый загруженный час:", max_hour)
print("Среднее количество аренд:", round(average_by_hour.max(), 2))
print("Самый малозагруженный час:", min_hour)
print("Среднее количество аренд:", round(average_by_hour.min(), 2))

print_section("ГРАФИК: СПРОС ПО ЧАСАМ")
plt.figure(figsize=(10, 5))
plt.plot(average_by_hour.index, average_by_hour.values, marker="o")
plt.title("Среднее количество арендованных велосипедов по часам")
plt.xlabel("Час")
plt.ylabel("Среднее количество аренд")
plt.xticks(range(24))
plt.grid(True)
save_current_figure("01_demand_by_hour.png")

# ============================================================
# 5. Анализ по сезонам
# ============================================================

print_section("АНАЛИЗ ПО СЕЗОНАМ")

seasonal = df.groupby("Seasons")["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)
print(seasonal)

average_by_season = df.groupby("Seasons")["Rented Bike Count"].mean()
print("\nСреднее по сезонам:")
print(average_by_season)

print_section("ГРАФИК: СПРОС ПО СЕЗОНАМ")
plt.figure(figsize=(8, 5))
plt.bar(average_by_season.index, average_by_season.values)
plt.title("Среднее количество арендованных велосипедов по сезонам")
plt.xlabel("Сезон")
plt.ylabel("Среднее количество аренд")
save_current_figure("02_demand_by_season.png")

# ============================================================
# 6. Работа и нерабочие дни системы
# ============================================================

print_section("РАБОЧИЕ И НЕРАБОЧИЕ ДНИ")

functioning = df.groupby("Functioning Day")["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)
print(functioning)

print("\nПроверка нулевого спроса при Functioning Day = No:")
non_functioning = df[df["Functioning Day"] == "No"]
print("Количество наблюдений:", len(non_functioning))
print("Уникальные значения Rented Bike Count:", non_functioning["Rented Bike Count"].unique())

print_section("ГРАФИК: РАБОЧИЕ И НЕРАБОЧИЕ ДНИ")
average_by_functioning = df.groupby("Functioning Day")["Rented Bike Count"].mean()
plt.figure(figsize=(7, 5))
plt.bar(average_by_functioning.index, average_by_functioning.values)
plt.title("Среднее количество аренд по типу дня работы системы")
plt.xlabel("Тип дня")
plt.ylabel("Среднее количество аренд")
save_current_figure("03_functioning_days.png")

# ============================================================
# 7. Праздники
# ============================================================

print_section("ПРАЗДНИКИ И ОБЫЧНЫЕ ДНИ")

holiday_analysis = df.groupby("Holiday")["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)
print(holiday_analysis)

print_section("ГРАФИК: ПРАЗДНИКИ")
average_by_holiday = df.groupby("Holiday")["Rented Bike Count"].mean()
plt.figure(figsize=(7, 5))
plt.bar(average_by_holiday.index, average_by_holiday.values)
plt.title("Среднее количество аренд в праздничные и обычные дни")
plt.xlabel("Тип дня")
plt.ylabel("Среднее количество аренд")
save_current_figure("04_holidays.png")

# ============================================================
# 8. Погодные показатели и корреляции
# ============================================================

print_section("ВЛИЯНИЕ ПОГОДНЫХ ПОКАЗАТЕЛЕЙ")

weather_columns = [
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)"
]

correlations = df[weather_columns + ["Rented Bike Count"]].corr()["Rented Bike Count"]
print("\nКорреляция с количеством арендованных велосипедов:")
for column in weather_columns:
    print(column + ":", round(correlations[column], 3))

correlation_ranking = (
    correlations.drop("Rented Bike Count")
    .sort_values(ascending=False)
)
print("\nРейтинг корреляций:")
print(correlation_ranking)

# ============================================================
# 9. Температурные диапазоны
# ============================================================

print_section("СПРОС В ЗАВИСИМОСТИ ОТ ТЕМПЕРАТУРЫ")

temperature_bins = [-100, 0, 10, 20, 30, 100]
temperature_labels = [
    "Ниже 0°C",
    "0–10°C",
    "10–20°C",
    "20–30°C",
    "30°C и выше"
]

df["Temperature Range"] = pd.cut(
    df["Temperature(°C)"],
    bins=temperature_bins,
    labels=temperature_labels,
    right=False
)

temperature_analysis = df.groupby(
    "Temperature Range",
    observed=False
)["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)
print(temperature_analysis)

# ============================================================
# 10. Дождь и снег
# ============================================================

print_section("ДОЖДЬ")
rain_analysis = df.groupby("Rain")["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)
print(rain_analysis)

print("\nСреднее количество аренд:")
for rain_status, value in df.groupby("Rain")["Rented Bike Count"].mean().items():
    print("С дождём:" if rain_status else "Без дождя:", round(value, 2))

print_section("СНЕГ")
snow_analysis = df.groupby("Snow")["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)
print(snow_analysis)

print("\nСреднее количество аренд:")
for snow_status, value in df.groupby("Snow")["Rented Bike Count"].mean().items():
    print("Со снегом:" if snow_status else "Без снега:", round(value, 2))

# ============================================================
# 11. Влажностные диапазоны
# ============================================================

print_section("СПРОС В ЗАВИСИМОСТИ ОТ ВЛАЖНОСТИ")

humidity_bins = [0, 30, 50, 70, 90, 101]
humidity_labels = [
    "0–30%",
    "30–50%",
    "50–70%",
    "70–90%",
    "90–100%"
]

df["Humidity Range"] = pd.cut(
    df["Humidity(%)"],
    bins=humidity_bins,
    labels=humidity_labels,
    right=False
)

humidity_analysis = df.groupby(
    "Humidity Range",
    observed=False
)["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max"]
)
print(humidity_analysis)

# ============================================================
# 12. Солнечная радиация
# ============================================================

print_section("СПРОС В ЗАВИСИМОСТИ ОТ СОЛНЕЧНОЙ РАДИАЦИИ")

solar_bins = [-0.01, 0, 0.5, 1, 2, 100]
solar_labels = [
    "0",
    "0–0.5",
    "0.5–1",
    "1–2",
    "2 и выше"
]

df["Solar Radiation Range"] = pd.cut(
    df["Solar Radiation (MJ/m2)"],
    bins=solar_bins,
    labels=solar_labels,
    right=False
)

solar_analysis = df.groupby(
    "Solar Radiation Range",
    observed=False
)["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max"]
)
print(solar_analysis)

# ============================================================
# 13. День недели и будни / выходные
# ============================================================

print_section("АНАЛИЗ ПО ДНЯМ НЕДЕЛИ")

weekday_analysis = df.groupby("Weekday")["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_analysis = weekday_analysis.reindex(weekday_order)
print(weekday_analysis)

print_section("БУДНИ И ВЫХОДНЫЕ")

day_type_analysis = df.groupby("Day Type")["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)
print(day_type_analysis)

print_section("ГРАФИК: БУДНИ И ВЫХОДНЫЕ")
average_by_day_type = df.groupby("Day Type")["Rented Bike Count"].mean()
plt.figure(figsize=(7, 5))
plt.bar(average_by_day_type.index, average_by_day_type.values)
plt.title("Среднее количество аренд: будни и выходные")
plt.xlabel("Тип дня")
plt.ylabel("Среднее количество аренд")
save_current_figure("05_weekday_weekend.png")

# ============================================================
# 14. Температура и спрос: точечный график
# ============================================================

print_section("ГРАФИК: ТЕМПЕРАТУРА И СПРОС")
plt.figure(figsize=(10, 5))
plt.scatter(
    df["Temperature(°C)"],
    df["Rented Bike Count"],
    alpha=0.3
)
plt.title("Зависимость количества аренд от температуры")
plt.xlabel("Температура (°C)")
plt.ylabel("Количество аренд")
save_current_figure("06_temperature_scatter.png")

# ============================================================
# 15. Общая статистика
# ============================================================

print_section("ОБЩАЯ СТАТИСТИКА")
rental_stats = df["Rented Bike Count"].agg(
    ["count", "mean", "median", "min", "max", "sum"]
)
print(rental_stats)

print_section("ИТОГОВАЯ СВОДКА")
print("Общее количество аренд:", int(df["Rented Bike Count"].sum()))
print("Среднее количество аренд за час:", round(df["Rented Bike Count"].mean(), 2))
print("Медианное количество аренд за час:", round(df["Rented Bike Count"].median(), 2))
print("Максимальное количество аренд за час:", int(df["Rented Bike Count"].max()))
print("Минимальное количество аренд за час:", int(df["Rented Bike Count"].min()))

# ============================================================
# 16. Контроль качества после подготовки
# ============================================================

print_section("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")

print("Пропуски:", df.isnull().sum().sum())
print("Полные дубликаты:", df.duplicated().sum())
print(
    "Дубликаты Дата + Час:",
    df.duplicated(subset=["Date", "Hour"]).sum()
)
print(
    "Диапазон часов:",
    df["Hour"].min(),
    "-",
    df["Hour"].max()
)
print("Количество уникальных дат:", df["Date"].nunique())

hours_per_day = df.groupby("Date")["Hour"].count()
print("Минимум записей за день:", hours_per_day.min())
print("Максимум записей за день:", hours_per_day.max())

numeric_columns = [
    "Rented Bike Count",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)"
]

negative_values = (df[numeric_columns] < 0).sum()
print("\nОтрицательные значения:")
print(negative_values)

print("\nАнализ завершён. Графики сохранены в папку 'figures'.")
