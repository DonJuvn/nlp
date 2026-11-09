# NLP-проект: Анализ текстов законов и нормативных актов (вариант 9)

Пошаговый план на сегодня — недели 1-4.

## 0. Открыть проект в VS Code

1. Открыть VS Code
2. `File -> Open Folder...` -> выбрать папку `nlp-legal-project`
3. Открыть терминал: `Terminal -> New Terminal` (или Ctrl+`)

## 1. Настройка окружения (один раз)

В терминале VS Code выполнить по очереди:

```bash
python -m venv venv
```

Активировать окружение:

```bash
# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows cmd
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

После активации в начале строки терминала появится `(venv)`. Дальше:

```bash
pip install -r requirements.txt
python -m spacy download ru_core_news_lg
```

Это займёт несколько минут (модель spaCy весит ~500 МБ).

## 2. Неделя 1 — сбор корпуса

Открыть `src/scraper.py`, заполнить словарь `SOURCES` (инструкция — в
комментарии в начале файла, там же 1 готовый рабочий пример).
Затем в терминале:

```bash
python src/scraper.py
```

Проверить результат:

```bash
python -c "import json; print(sum(1 for _ in open('data/raw/corpus.jsonl', encoding='utf-8')))"
```

Должно быть 500+ строк (документов-статей). Если меньше — добавьте
ещё кодексов в `SOURCES` и перезапустите.

## 3. Неделя 2 — предобработка

```bash
python src/preprocessing.py
```

Скрипт выведет в консоль 3 примера "до/после" и сохранит
`data/processed/corpus_clean.jsonl`.

## 4. Неделя 3 — морфология и лемматизация

```bash
python src/morphology_lemmatization.py
```

Выведет морфоразбор примера статьи, топ-10 слов/лемм/стемов и
сохранит сравнительную таблицу `report/lemma_vs_stem.csv`.

## 5. Неделя 4 — BOW и TF-IDF

```bash
python src/bow_tfidf.py
```

Сохранит `report/bow_top_words.csv`, `report/tfidf_top_words.csv`
и два графика `.png` — можно вставить прямо в отчёт.

## Что должно получиться к концу дня

- `data/raw/corpus.jsonl` — исходный корпус
- `data/processed/corpus_clean.jsonl` — очищенный + токенизированный
- `report/lemma_vs_stem.csv` — сравнение лемматизации/стемминга (неделя 3)
- `report/bow_top_words.csv`, `report/tfidf_top_words.csv` + графики (неделя 4)

Это закрывает недели 1-4 по учебному плану. Недели 5-7
(Word2Vec/GloVe/fastText, мини-задача, финальный отчёт) — отдельный
следующий шаг.
