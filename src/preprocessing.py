"""
Неделя 2 — Предобработка текста.

Запуск: python src/preprocessing.py
Вход:  data/raw/corpus.jsonl (создаётся scraper.py)
Выход: data/processed/corpus_clean.jsonl + отчёт до/после в консоль
"""

import json
import re
from pathlib import Path

import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "corpus.jsonl"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "corpus_clean.jsonl"

# Стандартные русские стоп-слова + кастомный список канцеляризмов,
# характерных именно для юридических текстов (важно для акцента проекта:
# обычный стоп-лист их не ловит, а они забивают частотный анализ)
RUSSIAN_STOPWORDS = set(stopwords.words("russian"))
LEGAL_STOPWORDS = {
    "настоящий", "настоящего", "настоящим", "настоящей",
    "указанный", "указанного", "указанной", "указанным",
    "вышеизложенный", "нижеследующий",
    "соответствии", "соответствующий", "соответствующим",
    "статья", "статьи", "статьей", "пункт", "пункта", "пункте",
    "часть", "части", "частью", "подпункт", "подпункта",
    "рк", "зрк", "кодекс", "кодекса", "кодексом",
}
ALL_STOPWORDS = RUSSIAN_STOPWORDS | LEGAL_STOPWORDS


def clean_text(text: str) -> str:
    """Убирает нумерацию статей/пунктов, спецсимволы, лишние пробелы."""
    text = re.sub(r"Статья\s+\d+(-\d+)?\.", " ", text)
    text = re.sub(r"\bп\.?\s?\d+(-\d+)?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\sёЁ-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text, language="russian")
    return [t for t in tokens if t.isalpha()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 2]


def preprocess_pipeline(raw_text: str) -> list[str]:
    """Полный пайплайн: очистка -> токенизация -> удаление стоп-слов."""
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return tokens


def load_corpus(path: Path) -> list[dict]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {RAW_PATH}. Сначала запустите src/scraper.py"
        )

    docs = load_corpus(RAW_PATH)
    print(f"Загружено документов: {len(docs)}")

    # демонстрация до/после на 3 примерах
    print("\n=== Сравнение до/после предобработки (3 примера) ===")
    for doc in docs[:3]:
        tokens = preprocess_pipeline(doc["text"])
        print(f"\n--- {doc['law']}, статья {doc['article_number']} ---")
        print("ДО:  ", doc["text"][:200].replace("\n", " "), "...")
        print("ПОСЛЕ:", " ".join(tokens[:30]), "...")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            tokens = preprocess_pipeline(doc["text"])
            out = {**doc, "tokens": tokens, "n_tokens": len(tokens)}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nСохранено -> {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
