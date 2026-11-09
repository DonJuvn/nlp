"""
Неделя 6 — Мини-задача: поиск похожих статей нормативных актов.

Задача: по текстовому запросу («увольнение работника по инициативе
работодателя») найти наиболее похожие статьи в корпусе. Делаем это
двумя способами и сравниваем их:
  1) TF-IDF + косинусное сходство — классический подход недели 4;
  2) усреднённые векторы fastText (неделя 5) + косинусное сходство —
     семантический подход, который умеет находить статьи по смыслу,
     даже если запрос не содержит точных слов из текста.

Плюс — визуализация корпуса: t-SNE проекция векторов документов,
раскрашенная по кодексу, чтобы увидеть, насколько хорошо статьи
разных кодексов разделяются в векторном пространстве (то есть
насколько вектора реально улавливают тематику).

Запуск: python src/mini_task_search.py
Вход:  data/processed/corpus_clean.jsonl   (метаданные: закон, статья, текст)
       data/processed/corpus_lemmas.jsonl  (леммы, из недели 5, тот же порядок строк)
       models/fasttext.model               (обучен в word_embeddings.py)
Выход: report/mini_task_search_results.csv
       report/mini_task_doc_clusters.png
"""

import csv
import json
from pathlib import Path

import numpy as np
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE_DIR / "data" / "processed" / "corpus_clean.jsonl"
LEMMAS_PATH = BASE_DIR / "data" / "processed" / "corpus_lemmas.jsonl"
MODEL_PATH = BASE_DIR / "models" / "fasttext.model"
REPORT_DIR = BASE_DIR / "report"
REPORT_DIR.mkdir(exist_ok=True)

TOP_N = 5

# Запросы для демонстрации мини-задачи — сформулированы обычным языком,
# а не точными терминами из кодекса, чтобы показать разницу TF-IDF vs fastText
DEMO_QUERIES = [
    "увольнение работника по инициативе работодателя",
    "уплата налога в бюджет",
    "расторжение брака между супругами",
    "наказание за кражу чужого имущества",
    "право собственности на земельный участок",
]


def load_meta_and_lemmas():
    """Читает метаданные документов и их леммы (строка в строку, один порядок)."""
    if not CLEAN_PATH.exists() or not LEMMAS_PATH.exists():
        raise FileNotFoundError(
            "Нужны data/processed/corpus_clean.jsonl и corpus_lemmas.jsonl. "
            "Сначала запустите preprocessing.py, затем word_embeddings.py "
            "(он создаёт кэш лемм)."
        )
    meta = []
    with CLEAN_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                meta.append({
                    "law": d["law"],
                    "article_number": d["article_number"],
                    "title": d.get("title", ""),
                    "text": d["text"],
                })
    lemmas = []
    with LEMMAS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lemmas.append(json.loads(line)["lemmas"])

    n = min(len(meta), len(lemmas))
    if len(meta) != len(lemmas):
        print(f"ВНИМАНИЕ: количество строк не совпадает ({len(meta)} vs {len(lemmas)}), "
              f"беру первые {n}")
    return meta[:n], lemmas[:n]


def build_tfidf_index(lemmas: list[list[str]]):
    texts = [" ".join(doc) for doc in lemmas]
    vectorizer = TfidfVectorizer(max_features=8000)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def build_fasttext_doc_vectors(model: FastText, lemmas: list[list[str]]) -> np.ndarray:
    """Вектор документа = среднее векторов его лемм (простой, но рабочий baseline)."""
    dim = model.wv.vector_size
    vectors = np.zeros((len(lemmas), dim), dtype=np.float32)
    for i, doc in enumerate(lemmas):
        word_vecs = [model.wv[w] for w in doc if w in model.wv]
        if word_vecs:
            vectors[i] = np.mean(word_vecs, axis=0)
    return vectors


def tfidf_query_vector(vectorizer, query: str, model: FastText):
    """Запрос лемматизируем тем же способом, что и корпус (через леммы fastText словаря
    не проверяем — TF-IDF использует свой словарь, поэтому просто нижний регистр + токены)."""
    return vectorizer.transform([query.lower()])


def fasttext_query_vector(model: FastText, query: str) -> np.ndarray:
    words = [w for w in query.lower().split() if w in model.wv]
    if not words:
        # fastText может построить вектор и для незнакомого слова через n-граммы
        words = query.lower().split()
    vecs = [model.wv[w] for w in words]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.wv.vector_size)


def search(query, meta, tfidf_vectorizer, tfidf_matrix, ft_model, ft_matrix, topn=TOP_N):
    tfidf_q = tfidf_query_vector(tfidf_vectorizer, query, ft_model)
    tfidf_sims = cosine_similarity(tfidf_q, tfidf_matrix)[0]
    tfidf_top = tfidf_sims.argsort()[::-1][:topn]

    ft_q = fasttext_query_vector(ft_model, query).reshape(1, -1)
    ft_sims = cosine_similarity(ft_q, ft_matrix)[0]
    ft_top = ft_sims.argsort()[::-1][:topn]

    return (
        [(meta[i], round(float(tfidf_sims[i]), 3)) for i in tfidf_top],
        [(meta[i], round(float(ft_sims[i]), 3)) for i in ft_top],
    )


def print_and_save_results(all_results, out_path):
    rows = [["query", "method", "rank", "law", "article", "title", "score"]]
    for query, (tfidf_res, ft_res) in all_results:
        print(f"\n=== Запрос: «{query}» ===")
        print("--- TF-IDF ---")
        for rank, (doc, score) in enumerate(tfidf_res, 1):
            print(f"  {rank}. [{score}] {doc['law']}, ст. {doc['article_number']} — {doc['title']}")
            rows.append(["query:" + query, "TF-IDF", rank, doc["law"], doc["article_number"], doc["title"], score])
        print("--- fastText ---")
        for rank, (doc, score) in enumerate(ft_res, 1):
            print(f"  {rank}. [{score}] {doc['law']}, ст. {doc['article_number']} — {doc['title']}")
            rows.append(["query:" + query, "fastText", rank, doc["law"], doc["article_number"], doc["title"], score])

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nРезультаты сохранены -> {out_path}")


def plot_clusters(meta, ft_matrix, out_path, sample_size=600, seed=42):
    """t-SNE проекция векторов документов, раскрашенная по названию закона."""
    rng = np.random.default_rng(seed)
    n = len(meta)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)

    coords = TSNE(
        n_components=2, random_state=seed, init="pca",
        perplexity=min(30, len(idx) // 4),
    ).fit_transform(ft_matrix[idx])

    laws = [meta[i]["law"] for i in idx]
    unique_laws = sorted(set(laws))
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(12, 9))
    for k, law in enumerate(unique_laws):
        mask = [l == law for l in laws]
        pts = coords[mask]
        plt.scatter(pts[:, 0], pts[:, 1], s=12, color=cmap(k % 10), label=law)
    plt.legend(fontsize=8, loc="best")
    plt.title("t-SNE проекция статей по векторам fastText (раскраска по кодексу)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"График кластеров сохранён -> {out_path}")


def main():
    meta, lemmas = load_meta_and_lemmas()
    print(f"Документов: {len(meta)}")

    print("Строю TF-IDF индекс...")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(lemmas)

    print("Загружаю модель fastText и строю вектора документов...")
    ft_model = FastText.load(str(MODEL_PATH))
    ft_matrix = build_fasttext_doc_vectors(ft_model, lemmas)

    all_results = []
    for query in DEMO_QUERIES:
        tfidf_res, ft_res = search(query, meta, tfidf_vectorizer, tfidf_matrix, ft_model, ft_matrix)
        all_results.append((query, (tfidf_res, ft_res)))

    print_and_save_results(all_results, REPORT_DIR / "mini_task_search_results.csv")
    plot_clusters(meta, ft_matrix, REPORT_DIR / "mini_task_doc_clusters.png")


if __name__ == "__main__":
    main()
