"""
Неделя 4 — Bag-of-Words и TF-IDF.

Запуск: python src/bow_tfidf.py
Вход:  data/processed/corpus_clean.jsonl
Выход: report/bow_top_words.csv, report/tfidf_top_words.csv,
       report/bow_top_words.png, report/tfidf_top_words.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "corpus_clean.jsonl"
REPORT_DIR = BASE_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_documents(path: Path) -> list[str]:
    """Каждая статья -> одна строка (токены через пробел) для векторайзера."""
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                texts.append(" ".join(doc["tokens"]))
    return texts


def top_n_from_matrix(matrix, feature_names, n=20):
    """Суммируем веса по всем документам и берём топ-n признаков."""
    scores = matrix.sum(axis=0).A1
    order = scores.argsort()[::-1][:n]
    return [(feature_names[i], round(float(scores[i]), 3)) for i in order]


def save_bar_chart(pairs: list[tuple], title: str, out_path: Path):
    words = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    plt.figure(figsize=(10, 8))
    plt.barh(words[::-1], values[::-1])
    plt.title(title)
    plt.xlabel("Суммарный вес")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_csv(pairs: list[tuple], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("word,score\n")
        for word, score in pairs:
            f.write(f"{word},{score}\n")


def main():
    texts = load_documents(PROCESSED_PATH)
    print(f"Документов для векторизации: {len(texts)}")

    # --- Bag-of-Words ---
    bow_vectorizer = CountVectorizer(max_features=5000)
    bow_matrix = bow_vectorizer.fit_transform(texts)
    bow_features = bow_vectorizer.get_feature_names_out()
    bow_top = top_n_from_matrix(bow_matrix, bow_features, n=20)

    print("\n=== Топ-20 слов по BOW (суммарная частота) ===")
    for word, score in bow_top:
        print(f"{word:<25}{score}")

    save_csv(bow_top, REPORT_DIR / "bow_top_words.csv")
    save_bar_chart(bow_top, "Топ-20 слов: Bag-of-Words", REPORT_DIR / "bow_top_words.png")

    # --- TF-IDF ---
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    tfidf_top = top_n_from_matrix(tfidf_matrix, tfidf_features, n=20)

    print("\n=== Топ-20 слов по TF-IDF (суммарный вес) ===")
    for word, score in tfidf_top:
        print(f"{word:<25}{score}")

    save_csv(tfidf_top, REPORT_DIR / "tfidf_top_words.csv")
    save_bar_chart(tfidf_top, "Топ-20 слов: TF-IDF", REPORT_DIR / "tfidf_top_words.png")

    print(f"\nВсё сохранено в {REPORT_DIR}")
    print(
        "\nВЫВОД ДЛЯ ОТЧЁТА: BOW обычно поднимает наверх общеслужебные "
        "юридические слова, повторяющиеся почти в каждой статье "
        "(они не попали в стоп-слова, но частотны везде). "
        "TF-IDF же понижает такие слова и поднимает термины, "
        "специфичные для конкретных статей/кодексов — сравните "
        "bow_top_words.csv и tfidf_top_words.csv и опишите разницу "
        "своими словами в итоговом отчёте."
    )


if __name__ == "__main__":
    main()
