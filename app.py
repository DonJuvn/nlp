"""
app.py - веб-приложение: определение кодекса РК по тексту статьи (неделя 5).

Запуск из корня проекта:
    pip install streamlit
    streamlit run app.py
Откроется страница http://localhost:8501
"""
import altair as alt
import pandas as pd
import streamlit as st

import predictor
from control_texts import CONTROL_TEXTS, law_name

LOW_CONF = 0.60   # ниже этой уверенности приложение предупреждает пользователя

TXT = {
    "ru": {
        "title": "Определение кодекса по тексту статьи",
        "sub": "Дообученная модель BERT по тексту статьи предсказывает, из какого кодекса РК она взята.",
        "input": "Текст статьи", "title_in": "Заголовок статьи (необязательно)",
        "analyze": "Анализировать", "pred": "Предсказанный кодекс", "conf": "Уверенность",
        "empty": "Сначала введите текст статьи.",
        "low": "Модель не уверена: вероятность ниже 60%. Возможно, это общая норма, которая встречается в нескольких кодексах.",
        "trunc": "Текст длинный: модель учла только первые {m} токенов из {n}.",
        "chart": "Вероятность по каждому кодексу", "law": "Кодекс", "p": "Вероятность, %",
        "history": "История запросов", "dl": "Скачать историю (CSV)", "ctrl": "Контрольные тексты",
        "none": "- выберите пример -", "expected": "Ожидаемый кодекс", "model": "Модель",
        "device": "Устройство", "maxlen": "Максимальная длина текста (токенов)",
        "spin": "Идёт анализ…", "nomodel": "Модель не найдена. Выполните ноутбук 4-й недели "
        "(папка models/bert_best) или 3-й недели (models/bert_legal_cls).",
        "hq": "Запрос", "hl": "Кодекс", "hp": "Уверенность, %", "ctrl_n": "Пример {i}",
    },
    "kk": {
        "title": "Мақала мәтіні бойынша кодексті анықтау",
        "sub": "Қосымша оқытылған BERT үлгісі мақаланың мәтіні бойынша оның ҚР-ның қай кодексінен алынғанын болжайды.",
        "input": "Мақала мәтіні", "title_in": "Мақала атауы (міндетті емес)",
        "analyze": "Талдау", "pred": "Болжанған кодекс", "conf": "Сенімділік",
        "empty": "Алдымен мақала мәтінін енгізіңіз.",
        "low": "Үлгі сенімсіз: ықтималдық 60 %-дан төмен. Бұл бірнеше кодексте кездесетін жалпы норма болуы мүмкін.",
        "trunc": "Мәтін ұзын: үлгі {n} токеннің тек алғашқы {m}-ін ескерді.",
        "chart": "Әр кодекс бойынша ықтималдық", "law": "Кодекс", "p": "Ықтималдық, %",
        "history": "Сұраныстар тарихы", "dl": "Тарихты жүктеу (CSV)", "ctrl": "Бақылау мәтіндері",
        "none": "- мысалды таңдаңыз -", "expected": "Күтілетін кодекс", "model": "Үлгі",
        "device": "Құрылғы", "maxlen": "Мәтіннің ең үлкен ұзындығы (токен)",
        "spin": "Талдау жүріп жатыр…", "nomodel": "Үлгі табылмады. 4-апта ноутбугын (models/bert_best) "
        "немесе 3-апта ноутбугын (models/bert_legal_cls) орындаңыз.",
        "hq": "Сұраныс", "hl": "Кодекс", "hp": "Сенімділік, %", "ctrl_n": "{i}-мысал",
    },
}

st.set_page_config(page_title="Кодекс по тексту статьи", layout="centered")

lang_choice = st.sidebar.radio("Язык / Тіл", ["Русский", "Қазақша"], key="lang_choice")
lang = "ru" if lang_choice == "Русский" else "kk"
T = TXT[lang]

model_dir = predictor.find_model_dir()
if model_dir is None:
    st.error(T["nomodel"])
    st.stop()


@st.cache_resource(show_spinner=False)
def get_predictor(path):
    return predictor.Predictor(path)


pred = get_predictor(model_dir)

st.session_state.setdefault("text", "")
st.session_state.setdefault("title", "")
st.session_state.setdefault("history", [])


def fill_example():
    i = st.session_state["ctrl"]
    if i > 0:
        st.session_state["text"] = CONTROL_TEXTS[i - 1]["text"]
        st.session_state["title"] = ""


with st.sidebar:
    st.selectbox(T["ctrl"], options=list(range(len(CONTROL_TEXTS) + 1)), key="ctrl",
                 format_func=lambda i: T["none"] if i == 0 else T["ctrl_n"].format(i=i),
                 on_change=fill_example)
    if st.session_state.get("ctrl", 0) > 0:
        exp = CONTROL_TEXTS[st.session_state["ctrl"] - 1]["expected"]
        st.caption(f"{T['expected']}: {law_name(exp, lang)}")
    st.divider()
    st.caption(f"{T['model']}: {pred.model_dir}")
    st.caption(f"{T['device']}: {pred.device}")
    st.caption(f"{T['maxlen']}: {pred.max_len}")

st.title(T["title"])
st.write(T["sub"])

st.text_area(T["input"], key="text", height=220)
st.text_input(T["title_in"], key="title")

if st.button(T["analyze"], type="primary"):
    text = st.session_state["text"]
    if not text.strip():
        st.warning(T["empty"])
    else:
        with st.spinner(T["spin"]):
            res = pred.predict(text, st.session_state["title"])
        ranked = res["ranked"]
        top_label, top_p = ranked[0]

        c1, c2 = st.columns([3, 1])
        c1.metric(T["pred"], law_name(top_label, lang))
        c2.metric(T["conf"], f"{top_p:.1%}")
        st.progress(min(max(top_p, 0.0), 1.0))
        if top_p < LOW_CONF:
            st.warning(T["low"])
        if res["truncated"]:
            st.info(T["trunc"].format(m=res["max_len"], n=res["n_tokens"]))

        df = pd.DataFrame({"law": [law_name(l, lang) for l, _ in ranked],
                           "p": [round(100 * p, 2) for _, p in ranked]})
        st.subheader(T["chart"])
        chart = (alt.Chart(df).mark_bar()
                 .encode(x=alt.X("p:Q", scale=alt.Scale(domain=[0, 100]), title=T["p"]),
                         y=alt.Y("law:N", sort=None, title=None),
                         tooltip=[alt.Tooltip("law:N", title=T["law"]),
                                  alt.Tooltip("p:Q", title=T["p"])])
                 .properties(width="container", height=34 * len(df)))
        st.altair_chart(chart)
        st.dataframe(df.rename(columns={"law": T["law"], "p": T["p"]}), hide_index=True)

        st.session_state["history"].append(
            {"q": " ".join(text.split())[:80], "l": law_name(top_label, lang),
             "p": round(100 * top_p, 1)})

if st.session_state["history"]:
    st.subheader(T["history"])
    hist = pd.DataFrame(st.session_state["history"]).rename(
        columns={"q": T["hq"], "l": T["hl"], "p": T["hp"]})
    st.dataframe(hist, hide_index=True)
    st.download_button(T["dl"], hist.to_csv(index=False).encode("utf-8-sig"),
                       file_name="history.csv", mime="text/csv")
