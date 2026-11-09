"""
Неделя 5 — Word2Vec, fastText и GloVe на корпусе нормативных актов РК.

Запуск: python src/word_embeddings.py
Вход:   data/processed/corpus_clean.jsonl (результат preprocessing.py)
Выход:  data/processed/corpus_lemmas.jsonl  (кэш лемм, считается один раз)
        models/word2vec.model, models/fasttext.model, models/glove_vectors.npz
        report/embeddings_neighbors.csv  (ближайшие слова во всех 3 моделях)
        report/embeddings_pairs.csv      (сходство юридических пар слов)
        report/word2vec_pca.png          (визуализация векторов)

Про GloVe: в Gensim нет обучения GloVe, а пакеты glove-python на
Windows почти не ставятся. Поэтому GloVe реализован здесь же на numpy
(оригинальный алгоритм Pennington et al. 2014: матрица совместной
встречаемости + взвешенная регрессия, оптимизатор AdaGrad).
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import FastText, Word2Vec
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "corpus_clean.jsonl"
LEMMAS_PATH = BASE_DIR / "data" / "processed" / "corpus_lemmas.jsonl"
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"
MODELS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Общие гиперпараметры — одинаковые для всех моделей, чтобы сравнение было честным
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 5
EPOCHS = 30
SEED = 42

# Слова для проверки ближайших соседей (юридические термины)
QUERY_WORDS = [
    "ответственность", "обязательство", "работодатель", "договор",
    "наказание", "собственность", "право", "суд", "налог", "брак",
]

# Пары, которые по смыслу должны быть близки (для количественного сравнения)
RELATED_PAIRS = [
    ("работник", "работодатель"), ("суд", "судья"),
    ("преступление", "наказание"), ("договор", "соглашение"),
    ("собственность", "имущество"), ("ответственность", "обязанность"),
    ("налог", "сбор"), ("брак", "супруг"), ("земля", "участок"),
    ("штраф", "взыскание"),
]


# ---------------------------------------------------------------- данные
def load_lemma_corpus() -> list[list[str]]:
    """Возвращает список документов, каждый — список лемм. Кэширует результат."""
    if LEMMAS_PATH.exists():
        print(f"Читаю кэш лемм: {LEMMAS_PATH}")
        with LEMMAS_PATH.open(encoding="utf-8") as f:
            return [json.loads(line)["lemmas"] for line in f if line.strip()]

    import spacy
    from preprocessing import ALL_STOPWORDS

    print("Кэша лемм нет — лемматизирую корпус через spaCy (один раз)...")
    nlp = spacy.load("ru_core_news_lg", disable=["parser", "ner"])
    with PROCESSED_PATH.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    texts = [" ".join(d["tokens"]) for d in docs]
    corpus = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=200)):
        lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
        lemmas = [l for l in lemmas if l not in ALL_STOPWORDS and len(l) > 2]
        corpus.append(lemmas)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(texts)}")

    with LEMMAS_PATH.open("w", encoding="utf-8") as f:
        for lemmas in corpus:
            f.write(json.dumps({"lemmas": lemmas}, ensure_ascii=False) + "\n")
    return corpus


# ---------------------------------------------------------------- GloVe
class GloVe:
    """Минимальная реализация GloVe на numpy (AdaGrad, sparse-матрица)."""

    def __init__(self, dim=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT,
                 x_max=100.0, alpha=0.75, lr=0.05, seed=SEED):
        self.dim, self.window, self.min_count = dim, window, min_count
        self.x_max, self.alpha, self.lr = x_max, alpha, lr
        self.rng = np.random.default_rng(seed)

    def _build_vocab(self, corpus):
        counts = Counter(w for doc in corpus for w in doc)
        words = [w for w, c in counts.most_common() if c >= self.min_count]
        self.index_to_key = words
        self.key_to_index = {w: i for i, w in enumerate(words)}

    def _cooccurrence(self, corpus):
        """Считает взвешенные (1/расстояние) совместные встречаемости в окне."""
        cooc = Counter()
        k2i = self.key_to_index
        for doc in corpus:
            ids = [k2i[w] for w in doc if w in k2i]
            for pos, wi in enumerate(ids):
                for off in range(1, self.window + 1):
                    if pos - off < 0:
                        break
                    wj = ids[pos - off]
                    cooc[(wi, wj)] += 1.0 / off
                    cooc[(wj, wi)] += 1.0 / off
        rows = np.fromiter((k[0] for k in cooc), dtype=np.int64, count=len(cooc))
        cols = np.fromiter((k[1] for k in cooc), dtype=np.int64, count=len(cooc))
        vals = np.fromiter(cooc.values(), dtype=np.float64, count=len(cooc))
        return rows, cols, vals

    def fit(self, corpus, epochs=EPOCHS, batch_size=8192):
        self._build_vocab(corpus)
        V, d = len(self.index_to_key), self.dim
        rows, cols, x = self._cooccurrence(corpus)
        print(f"  GloVe: словарь {V}, ненулевых пар {len(x)}")

        W = (self.rng.random((V, d)) - 0.5) / d
        C = (self.rng.random((V, d)) - 0.5) / d
        bw, bc = np.zeros(V), np.zeros(V)
        gW, gC = np.ones((V, d)), np.ones((V, d))
        gbw, gbc = np.ones(V), np.ones(V)

        weight = np.minimum(1.0, (x / self.x_max) ** self.alpha)
        logx = np.log(x)
        n = len(x)

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            total_cost = 0.0
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                i, j = rows[idx], cols[idx]
                wi, cj = W[i], C[j]
                diff = (wi * cj).sum(1) + bw[i] + bc[j] - logx[idx]
                fdiff = weight[idx] * diff
                total_cost += 0.5 * (weight[idx] * diff ** 2).sum()

                grad_w = fdiff[:, None] * cj
                grad_c = fdiff[:, None] * wi

                # AdaGrad: накапливаем квадраты градиентов, шаг делим на sqrt
                dW = np.zeros_like(W); np.add.at(dW, i, grad_w)
                dC = np.zeros_like(C); np.add.at(dC, j, grad_c)
                dbw = np.bincount(i, weights=fdiff, minlength=V)
                dbc = np.bincount(j, weights=fdiff, minlength=V)

                gW += dW ** 2; gC += dC ** 2
                gbw += dbw ** 2; gbc += dbc ** 2
                W -= self.lr * dW / np.sqrt(gW)
                C -= self.lr * dC / np.sqrt(gC)
                bw -= self.lr * dbw / np.sqrt(gbw)
                bc -= self.lr * dbc / np.sqrt(gbc)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  GloVe эпоха {epoch}/{epochs}, потери {total_cost / n:.4f}")

        # итоговый вектор слова = сумма основного и контекстного (как в статье)
        self.vectors = W + C
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.unit = self.vectors / np.maximum(norms, 1e-9)
        return self

    def __contains__(self, word):
        return word in self.key_to_index

    def similarity(self, a, b):
        return float(self.unit[self.key_to_index[a]] @ self.unit[self.key_to_index[b]])

    def most_similar(self, word, topn=5):
        sims = self.unit @ self.unit[self.key_to_index[word]]
        best = np.argsort(-sims)[1:topn + 1]
        return [(self.index_to_key[k], float(sims[k])) for k in best]

    def save(self, path):
        np.savez_compressed(path, vectors=self.vectors,
                            words=np.array(self.index_to_key))


# ---------------------------------------------------------------- сравнение
def neighbors_table(models: dict, topn=5):
    """Ближайшие слова к каждому запросу во всех моделях -> CSV + консоль."""
    rows = []
    for q in QUERY_WORDS:
        print(f"\n--- Ближайшие к «{q}» ---")
        for name, m in models.items():
            if q not in (m.wv if hasattr(m, "wv") else m):
                print(f"  {name:<9}: слова нет в словаре")
                continue
            res = m.wv.most_similar(q, topn=topn) if hasattr(m, "wv") else m.most_similar(q, topn)
            text = ", ".join(f"{w} ({s:.2f})" for w, s in res)
            print(f"  {name:<9}: {text}")
            rows.append([q, name, text])
    with (REPORT_DIR / "embeddings_neighbors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows([["query", "model", "neighbors"]] + rows)


def pairs_table(models: dict):
    """Среднее косинусное сходство «связанных» пар — грубая метрика качества."""
    header = ["pair"] + list(models)
    rows, sums, counts = [], Counter(), Counter()
    for a, b in RELATED_PAIRS:
        row = [f"{a} — {b}"]
        for name, m in models.items():
            kv = m.wv if hasattr(m, "wv") else m
            if a in kv and b in kv:
                s = kv.similarity(a, b)
                row.append(f"{s:.3f}")
                sums[name] += s; counts[name] += 1
            else:
                row.append("нет в словаре")
        rows.append(row)
    rows.append(["СРЕДНЕЕ"] + [f"{sums[n] / counts[n]:.3f}" if counts[n] else "-" for n in models])
    print("\n=== Сходство связанных юридических пар (косинус) ===")
    for r in [header] + rows:
        print("  " + " | ".join(f"{c:<28}" if i == 0 else f"{c:<14}" for i, c in enumerate(r)))
    with (REPORT_DIR / "embeddings_pairs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows([header] + rows)


def analogy_demo(w2v, ft):
    """Векторная арифметика: A - B + C. Результаты пояснительные, не гарантированные."""
    tests = [
        (["работодатель", "суд"], ["работник"]),
        (["преступление", "штраф"], ["наказание"]),
    ]
    print("\n=== Векторная арифметика (Word2Vec / fastText) ===")
    for pos, neg in tests:
        if not all(w in w2v.wv for w in pos + neg):
            continue
        for name, m in (("Word2Vec", w2v), ("fastText", ft)):
            res = m.wv.most_similar(positive=pos, negative=neg, topn=3)
            print(f"  {' + '.join(pos)} - {' - '.join(neg)} [{name}]: "
                  + ", ".join(f"{w} ({s:.2f})" for w, s in res))


def oov_demo(w2v, ft):
    """Главное отличие fastText: вектора для слов, которых не было в обучении."""
    print("\n=== Слова вне словаря (OOV): Word2Vec vs fastText ===")
    for w in ["правоприменительный", "работодателями", "недобросовестность"]:
        in_w2v = w in w2v.wv.key_to_index
        try:
            ft_res = ", ".join(x for x, _ in ft.wv.most_similar(w, topn=3))
        except KeyError:
            ft_res = "не удалось построить вектор"
        print(f"  «{w}»: Word2Vec — {'есть' if in_w2v else 'НЕТ вектора'}; fastText — {ft_res}")


def plot_pca(w2v):
    words = [w for w in QUERY_WORDS + [p for pair in RELATED_PAIRS for p in pair]
             if w in w2v.wv]
    words = list(dict.fromkeys(words))
    for q in QUERY_WORDS:
        if q in w2v.wv:
            words += [w for w, _ in w2v.wv.most_similar(q, topn=4)]
    words = list(dict.fromkeys(words))
    coords = PCA(n_components=2, random_state=SEED).fit_transform(
        np.array([w2v.wv[w] for w in words]))
    plt.figure(figsize=(12, 9))
    plt.scatter(coords[:, 0], coords[:, 1], s=15)
    for (x, y), w in zip(coords, words):
        plt.annotate(w, (x, y), fontsize=9)
    plt.title("Word2Vec: юридические термины (PCA, 2 компоненты)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "word2vec_pca.png", dpi=150)
    plt.close()


def main():
    corpus = load_lemma_corpus()
    n_tokens = sum(len(d) for d in corpus)
    print(f"Документов: {len(corpus)}, лемм: {n_tokens}")

    common = dict(vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT,
                  epochs=EPOCHS, seed=SEED, workers=1)  # workers=1 -> воспроизводимо

    t = time.time()
    print("\nОбучаю Word2Vec (skip-gram)...")
    w2v = Word2Vec(sentences=corpus, sg=1, **common)
    print(f"  готово за {time.time() - t:.0f} с, словарь {len(w2v.wv)}")

    t = time.time()
    print("Обучаю fastText (skip-gram, n-граммы 3-6)...")
    ft = FastText(sentences=corpus, sg=1, min_n=3, max_n=6, **common)
    print(f"  готово за {time.time() - t:.0f} с, словарь {len(ft.wv)}")

    t = time.time()
    print("Обучаю GloVe...")
    glove = GloVe().fit(corpus)
    print(f"  готово за {time.time() - t:.0f} с")

    w2v.save(str(MODELS_DIR / "word2vec.model"))
    ft.save(str(MODELS_DIR / "fasttext.model"))
    glove.save(MODELS_DIR / "glove_vectors.npz")

    models = {"Word2Vec": w2v, "fastText": ft, "GloVe": glove}
    neighbors_table(models)
    pairs_table(models)
    analogy_demo(w2v, ft)
    oov_demo(w2v, ft)
    plot_pca(w2v)
    print(f"\nМодели -> {MODELS_DIR}\nОтчёты и график -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
