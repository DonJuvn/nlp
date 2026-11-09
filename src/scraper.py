"""
Неделя 1 — Сбор корпуса нормативных правовых актов РК.

ВАЖНО: adilet.zan.kz рендерит текст через JavaScript, поэтому
requests.get() получает пустую заглушку "включите JavaScript".
Вместо него используем pavlodar.com/zakon/ — зеркало базы Минюста
РК с полностью статичным HTML (подтверждено вручную), содержащее
более 4300 документов в актуальных редакциях.

Страница вида https://pavlodar.com/zakon/?dok=<ID>&all=all отдаёт
ПОЛНЫЙ текст документа одним куском, разбитый на HTML-заголовки:
"Часть", "Раздел", "Глава", "Статья N. Название" — это и даёт
готовую разбивку на статьи (документы корпуса).

Список ниже — реальные ID (dok=...), собранные с главной страницы
https://pavlodar.com/zakon/ (актуальные редакции на 08.09.2026).
Можно добавить свои — там же на сайте, в подписи ссылки на каждый
документ виден его dok=XXXXX.
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SOURCES = {
    "Трудовой кодекс РК": "https://pavlodar.com/zakon/?dok=05995&all=all",
    "Уголовный кодекс РК": "https://pavlodar.com/zakon/?dok=05552&all=all",
    "Кодекс РК об административных правонарушениях": "https://pavlodar.com/zakon/?dok=05545&all=all",
    "Гражданский кодекс РК (Общая часть)": "https://pavlodar.com/zakon/?dok=00002&all=all",
    "Гражданский кодекс РК (Особенная часть)": "https://pavlodar.com/zakon/?dok=00001&all=all",
    "Земельный кодекс РК": "https://pavlodar.com/zakon/?dok=02847&all=all",
    "Экологический кодекс РК": "https://pavlodar.com/zakon/?dok=06850&all=all",
    "Кодекс РК о браке (супружестве) и семье": "https://pavlodar.com/zakon/?dok=04999&all=all",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# Заголовок статьи: "Статья 58." или "Статья 20-1."
ARTICLE_TITLE_RE = re.compile(r"^Статья\s+\d+(-\d+)?\.")


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def split_into_articles(html: str, law_name: str) -> list[dict]:
    """
    Проходит по документу и группирует текст под каждым заголовком
    "Статья N. ...". Любой другой заголовок (Часть/Раздел/Глава)
    просто закрывает текущую статью и не создаёт новый документ.
    """
    soup = BeautifulSoup(html, "html.parser")
    body_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"])

    articles = []
    current_title = None
    current_num = None
    current_parts: list[str] = []

    def flush():
        if current_title and current_parts:
            text = "\n".join(current_parts).strip()
            if text:
                articles.append(
                    {
                        "law": law_name,
                        "article_number": current_num,
                        "title": current_title,
                        "text": text,
                        "n_words": len(text.split()),
                    }
                )

    for tag in body_tags:
        text = tag.get_text(strip=True)
        if not text:
            continue
        is_heading = tag.name.startswith("h")
        if is_heading and ARTICLE_TITLE_RE.match(text):
            flush()
            m = re.match(r"^Статья\s+(\d+(?:-\d+)?)\.\s*(.*)", text)
            current_num, current_title = m.group(1), m.group(2)
            current_parts = []
        elif is_heading:
            flush()
            current_title = None
            current_parts = []
        else:
            if current_title:
                current_parts.append(text)

    flush()
    return articles


def build_corpus() -> list[dict]:
    corpus = []
    for law_name, url in SOURCES.items():
        print(f"Скачиваю: {law_name} ({url})")
        try:
            html = fetch_html(url)
        except requests.RequestException as e:
            print(f"  Ошибка загрузки: {e}")
            continue
        articles = split_into_articles(html, law_name)
        print(f"  Найдено статей: {len(articles)}")
        corpus.extend(articles)
        time.sleep(1)
    return corpus


def save_corpus(corpus: list[dict]) -> None:
    out_path = RAW_DIR / "corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"\nСохранено {len(corpus)} документов -> {out_path}")

    total_words = sum(doc["n_words"] for doc in corpus)
    print(f"Всего слов в корпусе: {total_words}")
    if len(corpus) < 500 and total_words < 50_000:
        print(
            "ВНИМАНИЕ: корпус меньше требуемого минимума. "
            "Добавьте ещё кодексов в SOURCES (см. https://pavlodar.com/zakon/)."
        )


if __name__ == "__main__":
    corpus = build_corpus()
    save_corpus(corpus)