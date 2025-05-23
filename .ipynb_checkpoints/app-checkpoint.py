import streamlit as st
import pickle
import re
import altair as alt
import pandas as pd


# —————— 1) SET PAGE CONFIG PRIMA DI OGNI ALTRA CHIAMATA st.* ——————
st.set_page_config(page_title="Sentiment & Watch Again", layout="centered")

# —————— 1) Funzione di pulizia del testo ——————
def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return s

@st.cache_data
def load_artifacts():
    # 1) Carica il vectorizer
    with open("vectorizer.pkl", "rb") as f:
        vect = pickle.load(f)

    # 2) Carica il modello Naive Bayes per il sentiment
    with open("nb_model.pkl", "rb") as f:
        clf_sent = pickle.load(f)

    # 3) Carica il modello Naive Bayes per watch-again
    with open("watch_model.pkl", "rb") as f:
        clf_watch = pickle.load(f)

    # 4) Carica i top‐token
    with open("tokens.pkl", "rb") as f:
        ts, ps, tw, pw = pickle.load(f)

    # *** IL RETURN DEVE ESSERE QUI, dentro la funzione ***
    return vect, clf_sent, clf_watch, ts, ps, tw, pw

# === Fuori dalla funzione, scarti i valori restituiti ===
vectorizer, clf_sent, clf_watch, ts, ps, tw, pw = load_artifacts()

labels_sent  = {0: "Negative 😞", 1: "Positive 😀"}
labels_watch = {0: "No Rewatch 🙁", 1: "Would Rewatch 😊"}

# —————— 3) Titoli e istruzioni ——————
st.title("📊 Sentiment & Rewatch Prediction")
st.markdown("📚 AI e ML per il marketing - IULM - Luca Tallarico 1034109")
st.write("⬇️ Inserisci una recensione e scopri sentiment e propensione a rivedere il film.")

# —————— 4) Input utente ——————
user_input = st.text_area("✏️ Testo della recensione", height=150)

if st.button("Analizza"):
    # — 1) Pulizia e vettorizzazione —
    clean = clean_text(user_input)
    X = vectorizer.transform([clean])

    # — 2) Predizioni —
    p_sent  = clf_sent.predict_proba(X)[0]
    pred_sent = clf_sent.predict(X)[0]
    p_watch = clf_watch.predict_proba(X)[0]
    pred_watch = clf_watch.predict(X)[0]

    # — 3) Output testuale —
    st.subheader("🎯 Sentiment")
    st.markdown(f"**Predizione:** {labels_sent[pred_sent]}")
    st.markdown(f"**Prob Positiva:** {p_sent[1]:.2f} | **Prob Negativa:** {p_sent[0]:.2f}")



    st.subheader("🎬 Watch Again")
    st.markdown(f"**Predizione:** {labels_watch[pred_watch]}")
    st.markdown(f"**Prob Rewatch:** {p_watch[1]:.2f} | **Prob No Rewatch:** {p_watch[0]:.2f}")


    # — 4) KPI metrics (USALO QUI, dentro il blocco!) —
    k1, k2, k3 = st.columns(3)
    k1.metric("Prob Positiva",    f"{p_sent[1]:.0%}", f"{(p_sent[1]- 
    p_sent[0]):.0%}")
    k2.metric("Prob Negativa",    f"{p_sent[0]:.0%}", None)
    k3.metric("Prob Rewatch",     f"{p_watch[1]:.0%}", f"{(p_watch[1]- 
    p_watch[0]):.0%}")

    # — 5) Bar-chart delle probabilità —
    df_sent  = pd.DataFrame({"Sentiment": ["Negative","Positive"], 
    "Probability": [p_sent[0], p_sent[1]]})
    df_watch = pd.DataFrame({"Watch Again": ["No","Yes"],        
    "Probability": [p_watch[0], p_watch[1]]})
    st.subheader("Probabilità:")
    st.subheader("📊 Probabilità Sentiment")
    st.bar_chart(df_sent.set_index("Sentiment"), use_container_width=True)
    st.subheader("🎬 Probabilità Watch Again")
    st.bar_chart(df_watch.set_index("Watch Again"), 
    use_container_width=True)

    # … e a seguire tutti gli altri grafici Altair, word‐cloud, ecc. …


       # —— 5) Top-10 Token Chart con Altair ——
    st.subheader("🔤 Top-10 Token")
    df_ts = pd.DataFrame({"Token": ts, "Log-Prob": ps})
    chart_sent = alt.Chart(df_ts).mark_bar().encode(
        x=alt.X("Log-Prob:Q", title="Log-Probability"),
        y=alt.Y("Token:N", sort="-x", title=None)
    ).properties(width=300, height=200,
                 title="Token più forti – Positive Sentiment")

    df_tw = pd.DataFrame({"Token": tw, "Log-Prob": pw})
    chart_watch = alt.Chart(df_tw).mark_bar(color="#FFAB00").encode(
        x="Log-Prob:Q",
        y=alt.Y("Token:N", sort="-x")
    ).properties(width=300, height=200,
                 title="Token più forti – Would Rewatch")

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_sent, use_container_width=True)
    with col2:
        st.altair_chart(chart_watch, use_container_width=True)




