import streamlit as st
import pickle
import re
import altair as alt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

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
# —————— dopo st.title(...) ——————
st.subheader("💡 Esempio recensioni - Copia e Incolla")
st.markdown(
    """
    **1)** I absolutely loved the movie, great performances, \
beautiful cinematography, and an unforgettable soundtrack.

    **2)** Good characters and a solid plot, but not something I’d revisit, \
one watch was enough.

    **3)** Poor script and over the top acting. I couldn’t connect with any \
of the characters.
    """
)

# —————— 4) Input utente ——————
user_input = st.text_area("✏️ Testo della recensione", height=150)

if st.button("Analizza"):

    # — Step 1: Mostro la recensione originale e l’input utente —
    st.subheader("1️⃣ Originale")
    st.write(user_input)

    # — Step 2: Pulizia del testo —
    st.subheader("2️⃣ Pulizia del testo")
    clean = clean_text(user_input)
    st.write("Cleaned text:", clean)

    # — Step 3: Vettorizzazione (TF-IDF) —
    st.subheader("3️⃣ Vettorizzazione con TF-IDF")
    X = vectorizer.transform([clean])
    # prendo solo i token presenti
    nz = X.nonzero()[1]
    present = [(vectorizer.get_feature_names_out()[i], float(X[0, i])) for i in nz]
    st.table(pd.DataFrame(present, columns=["Token", "TF-IDF"]))
    st.markdown("→ Abbiamo trasformato il testo in un vettore sparso, "\
                "dove ogni parola ha un peso TF-IDF.")

    # — 4️⃣ Breakdown Naive Bayes (solo sentiment, per brevità) —
    st.subheader("4️⃣ Breakdown log-posteriori (Sentiment)")
    rows = []
    for cls in (0,1):
        cum = clf_sent.class_log_prior_[cls]
        rows.append({"Classe": labels_sent[cls],
                     "Termine": "<PRIOR>",
                     "LogProb": f"{cum:.2f}",
                     "Cumulato": f"{cum:.2f}"})
        for tok, _ in present:
            idx = vectorizer.vocabulary_.get(tok)
            lp = clf_sent.feature_log_prob_[cls][idx]
            cum += lp
            rows.append({"Classe": "",
                         "Termine": tok,
                         "LogProb": f"{lp:.2f}",
                         "Cumulato": f"{cum:.2f}"})
    st.dataframe(pd.DataFrame(rows))

    # — 5️⃣ Predizione finale & KPI —
    st.subheader("5️⃣ Predizione e KPI")
    p_s = clf_sent.predict_proba(X)[0]
    p_w = clf_watch.predict_proba(X)[0]  # Assicurati che questa linea sia eseguita
    y_s, y_w = clf_sent.predict(X)[0], clf_watch.predict(X)[0]

    st.write(f"Debug - p_w: {p_w}")  # Aggiungi questa linea per il debugging

    st.success(f"🎯 Sentiment: **{labels_sent[y_s]}**  "
               f"(Pos={p_s[1]:.2f} | Neg={p_s[0]:.2f})")
    st.success(f"🎬 Watch Again: **{labels_watch[y_w]}**  "
               f"(Rewatch={p_w[1]:.2f} | No={p_w[0]:.2f})")
    c1,c2,c3 = st.columns(3)
    c1.metric("Prob Positiva", f"{p_s[1]:.0%}", f"{(p_s[1]-p_s[0]):.0%}")
    c2.metric("Prob Negativa", f"{p_s[0]:.0%}", None)
    c3.metric("Prob Rewatch",  f"{p_w[1]:.0%}", f"{(p_w[1]-p_w[0]):.0%}")

    # — 6️⃣ Grafico di riepilogo —
    st.subheader("6️⃣ Probabilità – grafico")
    df_s = pd.DataFrame({"Classe":["Neg","Pos"],"Prob":[p_s[0],p_s[1]]})
    df_w = pd.DataFrame({"No/Yes":["No","Yes"],"Prob":[p_w[0],p_w[1]]})

    import altair as alt

    # — 6️⃣ Probabilità – grafico e dietro le quinte —
    st.subheader("6️⃣ Probabilità – grafico e dietro le quinte")

    # 1) Costruisci i DataFrame delle probabilità
    prob_df_s = pd.DataFrame({
    "Classe":    ["Negative", "Positive"],
    "Probabilità": [p_s[0],    p_s[1]]
})
    prob_df_w = pd.DataFrame({
    "Watch Again": ["No Rewatch", "Would Rewatch"],
    "Probabilità": [p_w[0],     p_w[1]]
})

    # 2) Mostra le tabelle con i numeri esatti
    st.markdown("**Tabella delle probabilità (Sentiment)**")
    st.dataframe(prob_df_s)

    st.markdown("**Tabella delle probabilità (Watch Again)**")
    st.dataframe(prob_df_w)

    # 3) Versione Altair interattiva per il Sentiment
    st.markdown("**Grafico interattivo (Sentiment)**")
    chart_s = (
    alt.Chart(prob_df_s)
       .mark_bar()
       .encode(
           x=alt.X("Classe", title=None),
           y=alt.Y("Probabilità:Q", title="Probabilità"),
           tooltip=[
             alt.Tooltip("Classe", title="Classe"),
             alt.Tooltip("Probabilità:Q", title="Probabilità", format=".2f")
           ]
       )
       .properties(width=400, height=300)
       .interactive()
)
    st.altair_chart(chart_s, use_container_width=True)

    # 4) Versione Altair interattiva per il Watch Again
    st.markdown("**Grafico interattivo (Watch Again)**")
    chart_w = (
    alt.Chart(prob_df_w)
       .mark_bar()
       .encode(
           x=alt.X("Watch Again", title=None),
           y=alt.Y("Probabilità:Q", title="Probabilità"),
           tooltip=[
             alt.Tooltip("Watch Again", title="Watch Again"),
             alt.Tooltip("Probabilità:Q", title="Probabilità", format=".2f")
           ]
       )
       .properties(width=400, height=300)
       .interactive()
)
    st.altair_chart(chart_w, use_container_width=True)

    # 5) (Opzionale) Ricorda all’utente la soglia di decisione
    st.markdown(
    "_Per default classifichiamo “Positive” o “Would Rewatch” quando Prob ≥ 0.50_"
)

import pandas as pd

# … tuoi import, load_artifacts(), labels_sent/watch …

# Step 0: Anteprima del dataset
st.subheader("📋 Anteprima del dataset")
df = pd.read_csv("Synthetic Reviews.csv")

# Carico il CSV
df = pd.read_csv("Synthetic Reviews.csv")

# Se non esiste già, genero watch_again sinteticamente
if "watch_again" not in df.columns:
    import numpy as np
    df["watch_again"] = df["label"].apply(
        lambda l: int(np.random.rand() < (0.8 if l==1 else 0.1))
    )

# Mostro solo le prime 5 righe con text, label e watch_again
st.dataframe(df[["text","label","watch_again"]].head(5))
st.markdown("""
- **text**: la recensione testuale
- **label**: 0 = negativo, 1 = positivo
- **watch_again**: 0 = non rivedrebbe, 1 = rivedrebbe
""")

# —————— Valutazione globale del modello ——————
st.markdown("---")
st.subheader("📊 Metriche di valutazione sul dataset completo")

# Carico e pulisco tutto il dataset
df_full = pd.read_csv("Synthetic Reviews.csv")
if "watch_again" not in df_full.columns:
    df_full["watch_again"] = df_full["label"].apply(
        lambda l: int(np.random.rand() < (0.8 if l==1 else 0.1))
    )
df_full["clean"] = df_full["text"].apply(clean_text)

# Vettorizzo
X_full = vectorizer.transform(df_full["clean"])

# Predizioni sentiment
y_true = df_full["label"]
y_pred = clf_sent.predict(X_full)
y_prob = clf_sent.predict_proba(X_full)[:,1]

acc  = accuracy_score(y_true, y_pred)
auc  = roc_auc_score(y_true, y_prob)
cm   = confusion_matrix(y_true, y_pred)

# Mostro accuracy e AUC
c1, c2 = st.columns(2)
c1.metric("Accuracy Sentiment", f"{acc:.2%}")
c2.metric("ROC AUC Sentiment",   f"{auc:.2f}")

# Mostro confusion matrix
st.write("**Confusion Matrix (Sentiment)**")
st.dataframe(
    pd.DataFrame(
        cm,
        index=["Vero Neg","Vero Pos"],
        columns=["Pred Neg","Pred Pos"]
    )
)

# (Opzionale) stesse metriche per watch_again
y_true_w = df_full["watch_again"]
y_pred_w = clf_watch.predict(X_full)
y_prob_w = clf_watch.predict_proba(X_full)[:,1]

acc_w = accuracy_score(y_true_w, y_pred_w)
auc_w = roc_auc_score(y_true_w, y_prob_w)
cm_w  = confusion_matrix(y_true_w, y_pred_w)

c3, c4 = st.columns(2)
c3.metric("Accuracy WatchAgain", f"{acc_w:.2%}")
c4.metric("ROC AUC WatchAgain",   f"{auc_w:.2f}")

st.write("**Confusion Matrix (WatchAgain)**")
st.dataframe(
    pd.DataFrame(
        cm_w,
        index=["Vero No","Vero Sì"],
        columns=["Pred No","Pred Sì"]
    )
)

# —————— Mini glossario ——————
st.markdown("---")
st.subheader("📖 Mini-glossario")
st.markdown("""
- **TF-IDF**: peso di un termine _t_ nel documento _d_
  \\[
    \mathrm{tfidf}(t,d)
    = \frac{\mathrm{tf}(t,d)}{\sum_{t'} \mathrm{tf}(t',d)}
      \times \log\frac{N}{\mathrm{df}(t)}
  \\]

- **Log-prob**: \\(\log P(t\mid C)\\), peso del termine nel modello Naive Bayes.
- **Prior**: \\(\log P(C)\\), log-probabilità della classe sul training set.
- **Likelihood**: log-probabilità condizionata di ogni token, assunte indipendenti.
- **ROC AUC**: area sotto la curva ROC, misura la capacità di distinguere le classi.
""")

# — Naive Bayes Assumptions —
st.subheader("📚 Naive Bayes Classifier assume:")

# — Sezione di spiegazione dettagliata —
with st.expander("🧐 Come funziona il modello (clicca per espandere)"):
    st.markdown("""
    **1. Preprocessing**
    - Ogni recensione viene **pulita** (minuscolo, rimozione di punteggiatura e caratteri non alfabetici).
    - Viene trasformata in un **vettore TF-IDF**: ogni parola diventa una dimensione, il suo peso riflette frequenza e rarità.

    **2. Calcolo delle probabilità**
    - **Prior** \(P(C)\): la frequenza di ciascuna classe (positive/negative, rewatch/no) nel training set.
    
    - **Likelihood** \(P(x_i \mid C)\): il log-prob di ciascun token estratto dal modello (feature_log_prob_).
    
    - **Ipotesi di indipendenza**: si assume che i token siano condizionalmente indipendenti dato lo stato \(C\).

    **3. Formula di Bayes**
    \\[
      P(C \mid x)
        = \\frac{P(C) \\times \\prod_{i=1}^n P(x_i \\mid C)}{P(x)}
        \\propto P(C) \\times \\prod_{i=1}^n P(x_i \\mid C)
    \\]
    
    In *log-spazio* diventa:
    \\[
      \\log P(C \\mid x)
        = \\log P(C) + \\sum_{i=1}^n \\log P(x_i \\mid C)
    \\]

    **4. Decisione finale**
    - Calcoliamo il punteggio logaritmico per entrambe le classi.
    - Se \(\log P(\text{positive} \mid x) > \log P(\text{negative} \mid x)\), prediciamo *positive*, altrimenti *negative*.
    - Stessa logica per *watch_again* vs *no_watch_again*.

    **5. Vantaggi del Naive Bayes**
    - **Semplicità**: pochi parametri, addestra in millisecondi anche su decine di migliaia di esempi.
    
    - **Trasparenza**: puoi vedere direttamente i token più “forti” (quelli con log-prob più alti).
    
    - **Buone performance** su testi brevi e classificazione binaria.
    """)

# … qui prosegue il resto del tuo flusso (input, button, output, grafici) …

# … tutto il codice di prediction e grafici …

from sklearn.metrics import roc_curve

# ——————  X) Dietro le quinte della ROC sul test set ——————
st.header("📈 ROC curve sul test set")
# Calcola fpr, tpr e soglie per WatchAgain
fpr_w, tpr_w, thr_w = roc_curve(y_true_w, y_prob_w, pos_label=1)

# Costruisci il DataFrame
roc_df = pd.DataFrame({
    "threshold": thr_w,
    "false_positive_rate": fpr_w,
    "true_positive_rate": tpr_w
})

# Mostra i primi 10 punti in tabella
st.subheader("🔍 Tabella dei punti ROC")
st.dataframe(roc_df.head(10))

# Grafico interattivo con Altair
st.subheader("📈 Curva ROC (dietro le quinte)")
import altair as alt

roc_chart = (
    alt.Chart(roc_df)
       .mark_line(point=True)
       .encode(
           x=alt.X("false_positive_rate:Q", title="False Positive Rate"),
           y=alt.Y("true_positive_rate:Q",  title="True Positive Rate"),
           tooltip=[
               alt.Tooltip("threshold:Q", title="Threshold"),
               alt.Tooltip("false_positive_rate:Q", title="FPR"),
               alt.Tooltip("true_positive_rate:Q", title="TPR")
           ]
       )
       .properties(width=600, height=400)
       .interactive()
)
st.altair_chart(roc_chart, use_container_width=True)

# (Facoltativo) AUC di nuovo in calce
from sklearn.metrics import auc
auc_w2 = auc(fpr_w, tpr_w)
st.metric("🔢 AUC (calcolata manualmente)", f"{auc_w2:.3f}")

# ——————  X) Descrizione del progetto e algoritmo ——————
st.markdown("---")
st.subheader("📘 Descrizione del progetto")

st.markdown("""
Questo mini-tool serve a:
- **Classificare** in positivo/negativo una recensione testuale (sentiment analysis).
- **Stimare** la probabilità che un utente riveda (“watch again”) il film.

**Perché Naive Bayes?**
- È molto veloce da addestrare e predire (basta contare frequenze).
- Funziona bene su testi brevi, grazie all’ipotesi di indipendenza tra parole.
- Permette di ottenere direttamente **probabilità** per ogni classe.

---

### 🔎 Step-by-step dell’algoritmo Naive Bayes

1. **Tokenizzazione & TF-IDF**
   - Ogni recensione viene “pulita” (minuscolo, rimozione punteggiatura)
   - Costruiamo il vettore TF-IDF:
     \[
       \text{tfidf}_{i,j}
       = \frac{\text{tf}_{i,j}}{\sum_k \text{tf}_{k,j}}
       \times \log\frac{N}{\text{df}_i}
     \]
   - tf: frequenza del termine _i_ nel documento _j_; df: quante volte _i_ compare in qualsiasi doc; _N_ = numero totale di documenti.

2. **Stima delle probabilità a priori**
   - Calcolo \(P(\text{Positive})\) e \(P(\text{Negative})\) come proporzione di documenti di ciascuna classe.

3. **Stima delle probabilità condizionate**
   - Per ciascuna parola \(w\) e classe \(c\), calcolo
     \[
       P(w \mid c)
       = \frac{\text{conteggio}(w,c) + \alpha}{\sum_{w'} \text{conteggio}(w',c) + \alpha\,V}
     \]
     con _Laplace smoothing_ \(\alpha=1\), _V_ = dimensione del vocabolario.

4. **Predizione**
   - Dato un nuovo documento, calcolo per ciascuna classe \(c\):
     \[
       \log P(c \mid d)
       \;\propto\;
       \log P(c)
       + \sum_{w \in d} \log P(w\mid c)
     \]
   - Scelgo la classe con probabilità massima.

5. **Estensione “Watch Again”**
   - Allo stesso modo, addestro un NB secondario su un’etichetta sintetica watch_again (80% di rivedrebbe se sentiment positivo, 10% altrimenti).

---

🔧 **Architettura del codice**
- load_artifacts() carica:
  - vectorizer.pkl (TF-IDF)
  - nb_model.pkl (sentiment)
  - watch_model.pkl (watch_again)
  - tokens.pkl (top-10 token per classe)
- Alla pressione di **Analizza**, calcolo:
  1. Pulizia e vettorizzazione
  2. Predizioni + probabilità
  3. Visualizzazione testuale, bar-chart, token chart
  4. KPI (metriche percentuali)
""")
