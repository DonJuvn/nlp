"""
run_control_tests.py - прогон контрольных текстов через модель без интерфейса.
Результат: таблица в консоли и report/week5_control_tests.csv
(пригодится для отчёта рядом со скриншотами).

Запуск из корня проекта:  python run_control_tests.py
"""
import csv
import sys
from pathlib import Path

import predictor
from control_texts import CONTROL_TEXTS

model_dir = predictor.find_model_dir()
if model_dir is None:
    sys.exit("Модель не найдена. Выполните ноутбук 4-й недели (models/bert_best) "
             "или 3-й недели (models/bert_legal_cls).")

pred = predictor.Predictor(model_dir)
print(f"Модель: {model_dir}, устройство: {pred.device}\n")

rows, ok = [], 0
for i, c in enumerate(CONTROL_TEXTS, 1):
    label, p = pred.predict(c["text"])["ranked"][0]
    good = label == c["expected"]
    ok += good
    rows.append([i, c["expected"], label, f"{p:.4f}", "да" if good else "нет"])
    print(f"{i}. ожидалось: {c['expected']}\n   предсказано: {label} ({p:.1%}) - {'верно' if good else 'ОШИБКА'}")

print(f"\nВерно: {ok} из {len(rows)}")
Path("report").mkdir(exist_ok=True)
with open("report/week5_control_tests.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["№", "ожидаемый кодекс", "предсказанный кодекс", "уверенность", "верно"])
    w.writerows(rows)
print("Сохранено: report/week5_control_tests.csv")
