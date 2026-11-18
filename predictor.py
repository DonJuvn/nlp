"""
predictor.py - загрузка дообученной модели BERT и предсказание кодекса по тексту.
Используется и веб-приложением (app.py), и скриптом run_control_tests.py.
"""
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Модель ищется в таком порядке: переменная MODEL_DIR, лучшая модель недели 4,
# модель недели 3.
MODEL_CANDIDATES = ["models/bert_best", "models/bert_legal_cls"]


def find_model_dir():
    paths = ([os.environ["MODEL_DIR"]] if os.environ.get("MODEL_DIR") else []) + MODEL_CANDIDATES
    for p in paths:
        if (Path(p) / "config.json").exists():
            return p
    return None


def _read_max_len(default=256):
    """Длина текста при обучении (чтобы приложение резало текст так же)."""
    for name in ("report/week4_meta.json", "report/week3_hyperparameters.json"):
        try:
            return int(json.loads(Path(name).read_text(encoding="utf-8"))["max_len"])
        except (OSError, KeyError, ValueError):
            continue
    return default


class Predictor:
    def __init__(self, model_dir):
        self.model_dir = str(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_len = _read_max_len()

    def predict(self, text, title=""):
        """Возвращает кодексы с вероятностями (по убыванию) и сведения о длине текста."""
        title, text = title.strip(), text.strip()
        full = f"{title}. {text}" if title else text   # как при обучении: заголовок + текст
        n_tokens = len(self.tokenizer(full, truncation=False)["input_ids"])
        enc = self.tokenizer(full, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[0].float().cpu()
        probs = torch.softmax(logits, dim=-1).tolist()
        ranked = sorted(((self.id2label[i], p) for i, p in enumerate(probs)), key=lambda x: -x[1])
        return {"ranked": ranked, "n_tokens": n_tokens, "max_len": self.max_len,
                "truncated": n_tokens > self.max_len}
