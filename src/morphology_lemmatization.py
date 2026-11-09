"""
Неделя 3 — Морфологический анализ и сравнение spaCy-лемматизации
с NLTK-стеммингом (Snowball) на юридической лексике.

Установка модели перед запуском (один раз в терминале):
    python -m spacy download ru_core_news_lg

Запуск: python src/morphology_lemmatization.py
Вход:  data/processed/corpus_clean.jsonl
Выход: report/lemma_vs_stem.csv + топ-50 слов/лемм в консоль

ВАЖНО: лемматизация делается ПАЧКАМИ по документам через nlp.pipe(),
а не склейкой всего корпуса в одну строку — у spaCy жёсткий лимит
1 000 000 символов за один вызов (parser/NER съедают ~1ГБ памяти
на каждые 100k символов, весь корпус сразу просто не влезет).
Для лемматизации parser и NER не нужны — отключаем их, это заодно
сильно ускоряет обработку.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import spacy
from nltk.stem.snowball import SnowballStemmer

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "corpus_clean.jsonl"
REPORT_PATH = BASE_DIR / "report" / "lemma_vs_stem.csv"

# отключаем parser и ner — они не нужны для лемматизации/POS
# и являются основным источником потребления памяти
nlp = spacy.load("ru_core_news_lg", disable=["parser", "ner"])
stemmer = SnowballStemmer("russian")


def load_docs(path: Path) -> list[dict]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def morphological_analysis(text: str, n_tokens: int = 15) -> None:
    """Печатает POS/падеж/число для первых n_tokens слов текста."""
    doc = nlp(text[:2000])
    print(f"{'Слово':<20}{'Лемма':<20}{'POS':<8}{'Падеж':<10}{'Число'}")
    count = 0
    for token in doc:
        if not token.is_alpha:
            continue
        case = token.morph.get("Case")
        number = token.morph.get("Number")
        print(
            f"{token.text:<20}{token.lemma_:<20}{token.pos_:<8}"
            f"{','.join(case):<10}{','.join(number)}"
        )
        count += 1
        if count >= n_tokens:
            break


def lemmatize_documents(token_lists: list[list[str]], batch_size: int = 200) -> list[list[str]]:
    """
    Лемматизирует корпус документ-за-документом через nlp.pipe(),
    что не даёт превысить лимит длины текста и работает быстрее,
    чем вызывать nlp() по одному токену.
    """
    texts = [" ".join(tokens) for tokens in token_lists]
    result = []
    total = len(texts)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        result.append([t.lemma_ for t in doc if t.is_alpha])
        if (i + 1) % 500 == 0:
            print(f"  Лемматизировано документов: {i + 1}/{total}")
    return result


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stemmer.stem(t) for t in tokens]


def main():
    docs = load_docs(PROCESSED_PATH)
    print(f"Загружено документов: {len(docs)}\n")

    # 1. Морфоразбор на примере одной статьи
    print("=== Морфологический разбор (пример) ===")
    morphological_analysis(docs[0]["text"])

    # 2. Лемматизация (батчами по документам) vs стемминг (по токенам)
    token_lists = [doc["tokens"] for doc in docs]
    all_tokens: list[str] = [t for tokens in token_lists for t in tokens]
    print(f"\nВсего токенов в корпусе: {len(all_tokens)}")

    print("\nЛемматизация корпуса (батчами)...")
    lemma_lists = lemmatize_documents(token_lists)
    lemmas = [lemma for doc_lemmas in lemma_lists for lemma in doc_lemmas]

    print("Стемминг корпуса...")
    stems = stem_tokens(all_tokens)

    top_words = Counter(all_tokens).most_common(50)
    top_lemmas = Counter(lemmas).most_common(50)
    top_stems = Counter(stems).most_common(50)

    print("\n=== Топ-10 исходных слов vs лемм vs стемов ===")
    for i in range(10):
        w = top_words[i][0] if i < len(top_words) else ""
        l = top_lemmas[i][0] if i < len(top_lemmas) else ""
        s = top_stems[i][0] if i < len(top_stems) else ""
        print(f"{w:<20}{l:<20}{s}")

    # 3. Сохраняем сравнительную таблицу для отчёта
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "top_word", "count", "top_lemma", "count", "top_stem", "count"])
        for i in range(50):
            row = [i + 1]
            row += list(top_words[i]) if i < len(top_words) else ["", ""]
            row += list(top_lemmas[i]) if i < len(top_lemmas) else ["", ""]
            row += list(top_stems[i]) if i < len(top_stems) else ["", ""]
            writer.writerow(row)
    print(f"\nСравнительная таблица сохранена -> {REPORT_PATH}")

    # Пример проблемы стемминга на юр. терминах — для отчёта/выводов
    print("\n=== Демонстрация: где стемминг ломает юртермины ===")
    demo_words = ["правоприменение", "ответственность", "обязательство", "работодатель"]
    for w in demo_words:
        doc = nlp(w)
        lemma = doc[0].lemma_ if len(doc) else w
        stem = stemmer.stem(w)
        print(f"{w:<20} лемма: {lemma:<20} стем: {stem}")


if __name__ == "__main__":
    main()