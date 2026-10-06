# Seoul Bike Sharing Demand Analysis

Учебный проект по анализу набора данных **Seoul Bike Sharing Demand**.

## Состав репозитория

```text
SeoulBikeAnalysis/
├── Project_Brief_Seoul_Bike.md
├── exploratory_analysis.py
├── SeoulBikeData.csv
├── requirements.txt
└── figures/
```

### Назначение файлов

- `Project_Brief_Seoul_Bike.md` — постановка аналитической задачи, аналитические вопросы, требования к данным, верхнеуровневая оценка качества, ограничения и предварительные гипотезы.
- `exploratory_analysis.py` — детальный исследовательский анализ: расчёты показателей спроса, сравнения, корреляции и визуализации.
- `SeoulBikeData.csv` — исходный набор данных.
- `figures/` — графики, автоматически сохраняемые скриптом.

## Запуск

Требуется Python 3.x.

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

Запуск анализа:

```bash
python exploratory_analysis.py
```

После выполнения графики сохраняются в папку `figures`.

## Используемые библиотеки

- pandas
- matplotlib

## Набор данных

Seoul Bike Sharing Demand, UCI Machine Learning Repository, dataset 560.
