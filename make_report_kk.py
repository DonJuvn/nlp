"""
make_report_kk.py - собирает отчёт по 4-й неделе на казахском языке.

Запуск (из корня проекта, ПОСЛЕ того как отработал NLP_week4_experiments.ipynb):
    pip install python-docx
    python make_report_kk.py

Читает из папки report/ файлы, которые создаёт ноутбук:
    week4_experiments.csv, week4_all_errors.csv, week4_top_errors.csv,
    week4_meta.json, week4_experiments.png (график, необязательно)
Пишет: report/week4_report_kk.docx

Все цифры берутся из ваших результатов, ничего не придумывается.
Выводы формулируются по цифрам: при другом исходе эксперимента изменится
и текст заключения.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

REPORT = Path("report")
FONT = "Times New Roman"

# ---------------------------------------------------------------- словари
CAUSES_KK = {
    "шум в тексте (числа, таблицы, артефакты разметки)":
        "мәтіндегі шу (сандар, кестелер, белгілеу артефактілері)",
    "короткий малоинформативный текст": "қысқа, ақпараты аз мәтін",
    "обрезка длинного текста (часть статьи не попала в модель)":
        "ұзын мәтіннің қиылуы (мақаланың бір бөлігі үлгіге кірмеді)",
    "неоднозначность контекста (общая норма, встречается во многих кодексах)":
        "контекстің көп мағыналылығы (көптеген кодекстерде кездесетін жалпы норма)",
    "дисбаланс классов (редкий кодекс)": "сыныптар теңгерімсіздігі (сирек кездесетін кодекс)",
    "пересечение лексики смежных отраслей права":
        "сабақтас құқық салаларының лексикасының қиылысуы",
    "прочее": "басқа себептер",
}
FIXES_KK = {
    "мәтіндегі шу (сандар, кестелер, белгілеу артефактілері)":
        "мәтінді үлгіге бермес бұрын тазалау: кестелерді, сандық сілтемелерді және қызметтік таңбаларды алып тастау.",
    "қысқа, ақпараты аз мәтін":
        "мақалаға тарау атауын немесе көрші мақалаларды контекст ретінде қосу; өте қысқа мақалаларды «анықталмаған» деп белгілеу.",
    "ұзын мәтіннің қиылуы (мақаланың бір бөлігі үлгіге кірмеді)":
        "тек басын емес, мақаланың басы мен соңын алу немесе мәтінді үзінділерге бөліп, болжамдарды орташалау; бейнекартада max_len мәнін 512-ге дейін арттыру.",
    "контекстің көп мағыналылығы (көптеген кодекстерде кездесетін жалпы норма)":
        "мұндай нормалар («негізгі ұғымдар», «қағидаттар») барлық кодекстерде дерлік бірдей, бұл табиғи көп мағыналылық. «Жалпы норма» класын енгізуге немесе көрші мақалалардың контекстін ескеруге болады.",
    "сыныптар теңгерімсіздігі (сирек кездесетін кодекс)":
        "шығынның салмақталған функциясын (сынып салмақтарын) қолдану, сирек кодекстердің мысалдарын көбейту.",
    "сабақтас құқық салаларының лексикасының қиылысуы":
        "үлгіні ұзағырақ және көбірек мақалада оқыту, контекст белгілерін (бөлім, тарау) қосу, осы екі кодекс бойынша қателер матрицасын қарау.",
    "басқа себептер":
        "қолмен талдау: мүмкін, бұл ережеге қосуға тұрарлық жаңа себеп.",
}
LAWS_KK = {
    "Трудовой кодекс РК": "ҚР Еңбек кодексі",
    "Уголовный кодекс РК": "ҚР Қылмыстық кодексі",
    "Кодекс РК об административных правонарушениях": "ҚР Әкімшілік құқық бұзушылықтар туралы кодексі",
    "Гражданский кодекс РК": "ҚР Азаматтық кодексі",
    "Земельный кодекс РК": "ҚР Жер кодексі",
    "Экологический кодекс РК": "ҚР Экология кодексі",
    "Кодекс РК о браке (супружестве) и семье": "ҚР Неке (ерлі-зайыптылық) және отбасы туралы кодексі",
}


def law_kk(name):
    return LAWS_KK.get(name, name)


def cause_kk(name):
    return CAUSES_KK.get(name, name)


def article_kk(text):
    return re.sub(r"^ст\.\s*(\S+?)\.\s*", r"\1-бап. ", text)


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def lr_str(x):
    return f"{x:.1e}".replace(".", ",")


def pp(x):
    return f"{x:.1f}".replace(".", ",")


# ---------------------------------------------------------------- чтение
def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


need = ["week4_experiments.csv", "week4_all_errors.csv", "week4_top_errors.csv", "week4_meta.json"]
missing = [n for n in need if not (REPORT / n).exists()]
if missing:
    sys.exit("Не найдены файлы в report/: " + ", ".join(missing) +
             "\nСначала полностью выполните NLP_week4_experiments.ipynb (Run All).")

meta = json.loads((REPORT / "week4_meta.json").read_text(encoding="utf-8"))
exp = read_csv(REPORT / "week4_experiments.csv")
all_errors = read_csv(REPORT / "week4_all_errors.csv")
top_errors = read_csv(REPORT / "week4_top_errors.csv")

lr_rows, size_rows = exp[:3], exp[3:6]
f1 = lambda r: float(r["F1"])
acc = lambda r: float(r["Accuracy"])
best_by_val = max(lr_rows, key=lambda r: float(r["Val F1"]))
best_by_test = max(lr_rows, key=f1)
worst_by_test = min(lr_rows, key=f1)
lr_spread = 100 * (f1(best_by_test) - f1(worst_by_test))
best = size_rows[-1]                       # 100 % данных с лучшей скоростью
n_err, n_test = meta["n_errors"], meta["n_test"]

cause_counts = Counter(cause_kk(r["cause"]) for r in all_errors)
pair_counts = Counter((r["true_name"], r["pred_name"]) for r in all_errors)
size_delta = 100 * (f1(size_rows[-1]) - f1(size_rows[0]))
if size_delta > 0.5:
    size_trend, size_comment = "өсті", (
        "Бұл оқу деректері көбейген сайын үлгінің жақсарып жатқанын көрсетеді, яғни қосымша мақалалар қосу пайдалы.")
elif size_delta < -0.5:
    size_trend, size_comment = "төмендеді", (
        "Бұл күтпеген нәтиже: қосымша деректер сапаны жақсартпады. Ықтимал себеп — таңдамалардың шағын көлемі және "
        "бір seed-пен жүргізілген бір реттік эксперимент, сондықтан нәтижені қайта тексерген жөн.")
else:
    size_trend, size_comment = "іс жүзінде өзгермеді", (
        "Қосымша деректер айтарлықтай пайда бермеді: үлгі бұл есепті бар көлемде-ақ жақсы шешеді.")

# ---------------------------------------------------------------- документ
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin, sec.right_margin = Cm(3), Cm(1.5)
sec.top_margin = sec.bottom_margin = Cm(2)

st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(14)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)


def para(text="", bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True,
         size=14, after=6, keep=False):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing, pf.space_after, pf.space_before = 1.5, Pt(after), Pt(0)
    if indent:
        pf.first_line_indent = Cm(1.25)
    pf.keep_with_next = keep
    r = p.add_run(text)
    r.bold, r.italic, r.font.size, r.font.name = bold, italic, Pt(size), FONT
    return p


def heading(text, size=14):
    p = para(text, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, size=size, after=6, keep=True)
    p.paragraph_format.space_before = Pt(10)
    return p


def dash(text):
    p = para("— " + text, indent=False)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.space_after = Pt(3)
    return p


def table(head, rows, widths=None, size=11):
    t = doc.add_table(rows=1, cols=len(head))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(head):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size, r.font.name = True, Pt(size), FONT
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size, r.font.name = Pt(size), FONT
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# номер страницы внизу
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), "PAGE")
fp._p.append(fld)

# ---- титульный лист
for _ in range(6):
    para("", indent=False)
para("«Табиғи тілді өңдеу» пәні бойынша жоба жұмысы", align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
para("Тақырып: «Заңдар мен нормативтік құқықтық актілердің мәтіндерін талдау» (9-нұсқа)",
     align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
para("4-апта есебі: эксперименттік зерттеу және қателерді талдау", bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
for _ in range(7):
    para("", indent=False)
for line in ("Орындаған: ______________________", "Топ: ______________________",
             "Тексерген: ______________________"):
    para(line, align=WD_ALIGN_PARAGRAPH.RIGHT, indent=False, after=2)
for _ in range(5):
    para("", indent=False)
para("Астана, 2026", align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
doc.add_page_break()

# ---- 1. Кіріспе
heading("1. Кіріспе")
para("Бұл есепте «Табиғи тілді өңдеу» пәні бойынша жобаның 4-аптасында орындалған жұмыс баяндалған. "
     "3-аптада Қазақстан Республикасы кодекстерінің мәтіндері бойынша BERT үлгісі қосымша оқытылды "
     "(fine-tuning). 4-аптаның мақсаты — гиперпараметрлер мен оқу деректерінің көлемі үлгі сапасына қалай "
     "әсер ететінін зерттеу, метрикалардың толық жиынтығын есептеу, сондай-ақ үлгі қателескен мысалдарды "
     "талдап, қателердің себептерін анықтау және оларды түзету жолдарын ұсыну.")
para("Жобаның тақырыбы мен деректер жиыны 1-аралық бақылаудан өзгеріссіз сақталды: бұл Қазақстан "
     "Республикасының сегіз кодексінен алынған заң мақалалары.")

# ---- 2. Есеп және деректер
heading("2. Есептің қойылымы және деректер")
para("Жіктеу есебі шешілді: мақаланың мәтіні бойынша ол қай кодекске жататынын анықтау керек. Белгі "
     "корпустың өзінен алынады: әр мақала қай кодекстен алынғанын біледі, сондықтан қолмен белгілеу қажет "
     f"болмады. Азаматтық кодекстің жалпы және ерекше бөліктері бір сыныпқа біріктірілді, өйткені жеке мақала "
     f"бойынша оларды ажырату мағынасыз. Нәтижесінде {len(meta['labels'])} сынып болды.")
para(f"Деректер үш бөлікке бөлінді: оқу ({meta['n_train']} мақала), валидациялық ({meta['n_val']}) және "
     f"сынақ ({meta['n_test']}) таңдамалары. Сынақ таңдамасы 3-аптадағымен бірдей (seed 42), сыныптар бойынша "
     "стратификациямен жасалды. Валидациялық таңдама оқу бөлігінің 10 %-ы және ең жақсы оқу жылдамдығын "
     "таңдау үшін ғана қолданылды. Сынақ таңдамасы тек қорытынды сандар мен қателерді талдау үшін "
     "пайдаланылды. Бұл деректердің жылыстауының алдын алады.")

# ---- 3. Әдістеме
heading("3. Эксперименттер әдістемесі")
para(f"Барлық эксперименттерде бірдей параметрлер қолданылды: үлгі — {meta['model']}; оқу дәуірлерінің "
     f"саны — {meta['epochs']}; batch өлшемі — {meta['batch_size']}; мәтіннің ең үлкен ұзындығы — "
     f"{meta['max_len']} токен (одан ұзын мәтін қиылады); оңтайландырғыш — AdamW, салмақтың ыдырауы — 0,01; "
     "оқу жылдамдығы алғашқы 10 % қадамда біртіндеп өсіп, кейін нөлге дейін сызықты төмендейді (warmup); "
     "кездейсоқ сандар генераторының бастапқы мәні (seed) — 42. Тек зерттелетін параметр өзгертілді.")
para("Әр эксперимент алдын ала оқытылған салмақтардан басталды. 3-аптада қосымша оқытылған үлгі бастапқы "
     "нүкте ретінде алынбады: ол бұл деректерді көріп қойған, сондықтан оқу жылдамдықтарын әділ "
     "салыстыруға болмас еді. 3-аптаның параметрлері торға орталық нүкте ретінде кірді.")
lrs = [float(r["lr"]) for r in lr_rows]
para(f"1-эксперимент — оқу жылдамдығы. Үш мән салыстырылды: {lr_str(lrs[0])}, {lr_str(lrs[1])} және "
     f"{lr_str(lrs[2])} (3-аптадағы мәннің үштен бірі, өзі және 5/3 бөлігі). Деректердің 100 %-ы "
     "қолданылды. Ең жақсы мән валидациялық таңдамадағы F1 бойынша таңдалды.")
para("2-эксперимент — оқу таңдамасының көлемі. Ең жақсы оқу жылдамдығымен үлгі оқу деректерінің 50 %, 75 % "
     "және 100 %-ында оқытылды. Ішкі таңдама сыныптар бойынша стратификацияланып алынды.")
para("Сапа төрт метрикамен бағаланды. Accuracy (дәлдік) — дұрыс болжамдардың үлесі. Precision (нақтылық) — "
     "үлгі белгілеген мақалалардың қаншасы шын мәнінде сол кодекске жататынын көрсетеді. Recall (толықтық) — "
     "кодекстің нақты мақалаларының қаншасы табылғанын көрсетеді. F1 — нақтылық пен толықтықтың "
     "гармониялық ортасы. Precision, Recall және F1 сыныптар бойынша macro-орташаланды: әр кодекс бірдей "
     "салмақпен есептеледі, сондықтан шағын кодекстердегі нәтиже де жасырылмайды.")

# ---- 4. Нәтижелер
heading("4. Эксперименттер нәтижелері")
heading("4.1. Оқу жылдамдығының әсері")
para("1-кестеде оқу жылдамдығының үш мәні үшін сынақ таңдамасындағы метрикалар және валидациядағы F1 "
     "берілген.", keep=True)
para("1-кесте. Оқу жылдамдығының әсері (оқу деректерінің 100 %-ы)", indent=False,
     align=WD_ALIGN_PARAGRAPH.LEFT, size=12, after=2, keep=True)
table(["Оқу жылдамдығы", "Val F1", "Accuracy", "Precision", "Recall", "F1"],
      [[lr_str(float(r["lr"])), num(float(r["Val F1"]), 4), num(acc(r), 4), num(float(r["Precision"]), 4),
        num(float(r["Recall"]), 4), num(f1(r), 4)] for r in lr_rows])
para(f"Валидациядағы F1 бойынша ең жақсы мән ретінде {lr_str(float(best_by_val['lr']))} таңдалды. "
     f"Сынақ таңдамасында ең жоғары F1 — {num(f1(best_by_test), 4)} ({lr_str(float(best_by_test['lr']))}), "
     f"ең төмені — {num(f1(worst_by_test), 4)} ({lr_str(float(worst_by_test['lr']))}); айырма "
     f"{pp(lr_spread)} п.п.")
if best_by_val is not best_by_test:
    para("Валидация бойынша таңдалған жылдамдық сынақ таңдамасында ең жоғары нәтиже бермеді. Бұл "
         "валидациялық таңдама шағын болғандықтан кездейсоқ ауытқумен түсіндірілуі мүмкін. Дегенмен таңдау "
         "сынақ таңдамасы бойынша емес, валидация бойынша жасалды: сынақ деректері үлгіні таңдауға "
         "қатыспайды.")

heading("4.2. Оқу таңдамасы көлемінің әсері")
para("2-кестеде ең жақсы оқу жылдамдығымен, оқу деректерінің әртүрлі үлесінде алынған нәтижелер берілген.",
     keep=True)
para("2-кесте. Оқу деректерінің көлемі мен сапа", indent=False, align=WD_ALIGN_PARAGRAPH.LEFT,
     size=12, after=2, keep=True)
table(["Деректер үлесі", "Оқу мақалалары", "Accuracy", "Precision", "Recall", "F1"],
      [[r["Доля данных"], r["Обучающих статей"], num(acc(r), 4), num(float(r["Precision"]), 4),
        num(float(r["Recall"]), 4), num(f1(r), 4)] for r in size_rows])
para(f"Оқу деректері 50 %-дан 100 %-ға дейін ұлғайғанда F1 {num(f1(size_rows[0]), 4)} мәнінен "
     f"{num(f1(size_rows[-1]), 4)} мәніне дейін {size_trend} (өзгеріс {pp(size_delta)} п.п.). {size_comment}")

heading("4.3. Метрикалар динамикасы")
png = REPORT / "week4_experiments.png"
if png.exists():
    para("1-суретте оқу жылдамдығы мен деректер көлемінің сапаға әсері, сондай-ақ дәуірлер бойынша F1 "
         "өзгерісі көрсетілген.", keep=True)
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.add_run().add_picture(str(png), width=Cm(15.5))
    para("1-сурет. Эксперименттер нәтижелері: оқу жылдамдығының әсері, F1 динамикасы, деректер көлемінің әсері",
         indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
else:
    para("Графиктер report/week4_experiments.png файлында сақталған.")

# ---- 5. Қателерді талдау
heading("5. Қателерді талдау")
para(f"Ең жақсы үлгі ({lr_str(float(best['lr']))}, деректердің 100 %-ы) сынақ таңдамасындағы {n_test} "
     f"мақаланың {n_err}-інде қателесті (дәлдік {num(acc(best), 4)}). Әр қатеге негізгі себеп автоматты "
     "ережелер бойынша берілді. Ережелер ретімен тексеріледі, бірінші орындалғаны себепті анықтайды.")
heading("5.1. Қателер себептері")
para("3-кестеде қателердің себептер бойынша бөлінуі берілген.", keep=True)
para("3-кесте. Қателер себептері", indent=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=12, after=2, keep=True)
total_err = sum(cause_counts.values()) or 1
table(["Себеп", "Қателер саны", "Үлесі, %"],
      [[c, n, pp(100 * n / total_err)] for c, n in cause_counts.most_common()], widths=[10.5, 3, 2.5])
if pair_counts:
    pairs = pair_counts.most_common(3)
    txt = "; ".join(f"«{law_kk(t)}» және «{law_kk(p)}» ({n} рет)" for (t, p), n in pairs)
    para(f"Ең жиі шатастырылған кодекстер: {txt}.")
para("Ескерту. Ресми кодекстерде қате терілген сөздер іс жүзінде кездеспейді, сондықтан «қате терілген сөз» "
     "себебінің орнына мәтіндегі шу (сандар, кестелер, белгілеу артефактілері) енгізілді. Белгі қолмен "
     "қойылмай, мақаланың шыққан кодексінен алынғандықтан, белгілеу қателері бұл есепте жоқ.")

heading("5.2. Қателер мысалдары")
para("4-кестеде үлгі қателескен мақалалардың мысалдары берілген (толық тізім — report/week4_top_errors.csv "
     "файлында, онда мәтін үзінділері де бар).", keep=True)
para("4-кесте. Үлгі қателескен мақалалар", indent=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=12, after=2,
     keep=True)
rows = []
for k, r in enumerate(top_errors[:15], 1):
    rows.append([k, law_kk(r["Кодекс (верно)"]), law_kk(r["Предсказано"]), num(float(r["Уверенность"]), 2),
                 article_kk(r["Статья"]), cause_kk(r["Причина"])])
table(["№", "Нақты кодекс", "Болжам", "Сенімділік", "Мақала", "Себеп"], rows,
      widths=[0.9, 3.0, 3.0, 1.8, 4.0, 3.3], size=9)

heading("5.3. Түзету жолдары")
para("Анықталған себептердің әрқайсысы үшін төмендегі шешімдер ұсынылады:", keep=True)
for c, _ in cause_counts.most_common():
    dash(f"{c}: {FIXES_KK.get(c, '')}")

# ---- 6. Қорытынды
heading("6. Қорытынды")
para(f"Жұмыс барысында BERT үлгісі ({meta['model']}) заң мақаласының мәтіні бойынша оның қай кодекске "
     "жататынын анықтау есебінде қосымша оқытылып, оның сапасына екі фактордың әсері зерттелді: оқу "
     "жылдамдығы және оқу деректерінің көлемі.")
if lr_spread < 1.0:
    lr_concl = ("Айырма 1 п.п.-тен аз, яғни үш мән де шамамен бірдей нәтиже берді. Бұл жағдайда оқу "
                "жылдамдығын таңдау сапаға аз әсер етеді.")
else:
    lr_concl = "Бұл оқу жылдамдығын таңдау нәтижеге байқалатын әсер ететінін көрсетеді."
para("Оқу жылдамдығы бойынша: " + ", ".join(f"{lr_str(float(r['lr']))} үшін F1 = {num(f1(r), 3)}"
                                           for r in lr_rows) +
     f". Ең жоғары және ең төмен нәтиженің айырмасы {pp(lr_spread)} п.п. {lr_concl}")
para(f"Оқу деректерінің көлемі бойынша: деректер 50 %-дан 100 %-ға дейін ұлғайғанда F1 {size_trend} "
     f"({num(f1(size_rows[0]), 3)} → {num(f1(size_rows[-1]), 3)}). {size_comment}")
top_cause, top_n = cause_counts.most_common(1)[0] if cause_counts else ("-", 0)
para(f"Ең жақсы үлгі сынақ таңдамасында дәлдікке {num(acc(best), 3)} және F1 = {num(f1(best), 3)} жетті; "
     f"{n_test} мақаланың {n_err}-інде қателесті. Қателердің ең жиі себебі — {top_cause} ({top_n} қате). "
     f"Бұл себеп үшін ұсынылатын шешім: {FIXES_KK.get(top_cause, '')}")
para(f"Нәтижелерді қорытындылағанда мыналарды ескеру қажет: эксперименттер бір seed-пен (42) бір рет "
     f"орындалды, сынақ таңдамасы шағын ({n_test} мақала), сондықтан 1–2 п.п.-ке дейінгі айырмашылықтар "
     "кездейсоқ ауытқу шегінде болуы мүмкін. Қателер себептері автоматты ережелер бойынша анықталды, "
     f"олардың шектері (мысалы, қысқа мәтін — {meta['short_words']} сөзден аз) шартты.")
para("Келесі аптада (5-апта) ең жақсы үлгі негізінде Streamlit немесе Gradio көмегімен веб-қосымша "
     "әзірленеді: онда пайдаланушы мақаланың мәтінін енгізіп, болжанған кодекс пен үлгінің сенімділігін "
     "көреді.")

out = REPORT / "week4_report_kk.docx"
doc.save(out)
print("Готово:", out)
