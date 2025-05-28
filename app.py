import streamlit as st
import pickle
import re
import altair as alt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc

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

    return vect, clf_sent, clf_watch, ts, ps, tw, pw

# === Caricamento artefatti ===
vectorizer, clf_sent, clf_watch, ts, ps, tw, pw = load_artifacts()

labels_sent  = {0: "Negative 😞", 1: "Positive 😀"}
labels_watch = {0: "No Rewatch 🙁", 1: "Would Rewatch 😊"}

# —————— HEADER E DESCRIZIONE PROGETTO ——————
st.title("🤖 Sentiment & Rewatch Prediction")
st.markdown("📚 **AI e ML per il marketing - IULM - Luca Tallarico 1034109**")

st.markdown("""
### 🎯 **Obiettivo del progetto**
Questo strumento dimostra il funzionamento **interno** del classificatore Naive Bayes attraverso:
- 📊 **Analisi step-by-step** di come il modello elabora il testo
- 🔢 **Calcoli numerici espliciti** con formule matematiche
- 🧮 **Confronto manuale vs automatico** per verificare la correttezza
- 📈 **Visualizzazioni interattive** dei risultati
""")

# —————— ANTEPRIMA DATASET ——————
st.subheader("📋 Anteprima del dataset di training")

# Carico il CSV
df = pd.read_csv("Synthetic Reviews.csv")

# Se non esiste già, genero watch_again sinteticamente
if "watch_again" not in df.columns:
    np.random.seed(42)  # Per riproducibilità
    df["watch_again"] = df["label"].apply(
        lambda l: int(np.random.rand() < (0.8 if l==1 else 0.1))
    )

# Mostro statistiche del dataset
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📝 Totale recensioni", len(df))
with col2:
    pos_count = (df["label"] == 1).sum()
    st.metric("😀 Recensioni positive", f"{pos_count} ({pos_count/len(df):.1%})")
with col3:
    rewatch_count = (df["watch_again"] == 1).sum()
    st.metric("🎬 Rivedrebbero", f"{rewatch_count} ({rewatch_count/len(df):.1%})")

# Mostra esempi
st.dataframe(df[["text","label","watch_again"]].head(5))
st.markdown("""
- **text**: la recensione testuale originale
- **label**: 0 = negativo, 1 = positivo  
- **watch_again**: 0 = non rivedrebbe, 1 = rivedrebbe (generato sinteticamente)
""")

# —————— ESEMPI DI RECENSIONI ——————
st.subheader("💡 Esempi di recensioni - Prova questi!")

examples = [
    "I absolutely loved the movie, great performances, beautiful cinematography, and an unforgettable soundtrack.",
    "Good characters and a solid plot, but not something I'd revisit, one watch was enough.",
    "Poor script and over the top acting. I couldn't connect with any of the characters.",
    "Amazing visual effects and stellar acting made this an incredible experience worth watching again.",
    "Boring and predictable storyline. Waste of time and money."
]

selected_example = st.selectbox("Scegli un esempio:", ["Scrivi la tua..."] + examples)

# —————— INPUT UTENTE ——————
if selected_example == "Scrivi la tua...":
    user_input = st.text_area("✏️ Inserisci la tua recensione:", height=100)
else:
    user_input = st.text_area("✏️ Recensione selezionata (puoi modificarla):", 
                             value=selected_example, height=100)

# —————— ANALISI PRINCIPALE ——————
if st.button("🔍 Analizza Recensione", type="primary"):
    if not user_input.strip():
        st.error("⚠️ Inserisci una recensione per procedere!")
        st.stop()

    # ═══════════════════════════════════════════════════════════
    # SEZIONE 1: PREPROCESSING
    # ═══════════════════════════════════════════════════════════
    st.header("📝 **FASE 1: Preprocessing del Testo**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔤 Testo Originale")
        st.info(f'"{user_input}"')
        st.caption(f"Lunghezza: {len(user_input)} caratteri")
    
    with col2:
        clean = clean_text(user_input)
        st.subheader("🧹 Testo Pulito")
        st.success(f'"{clean}"')
        st.caption(f"Lunghezza: {len(clean)} caratteri")
    
    st.markdown("**🔧 Operazioni di pulizia:**")
    st.markdown("1. Conversione in minuscolo: `text.lower()`")
    st.markdown("2. Rimozione caratteri non alfabetici: `re.sub(r'[^a-z\\s]', '', text)`")

    # ═══════════════════════════════════════════════════════════
    # SEZIONE 2: VETTORIZZAZIONE TF-IDF
    # ═══════════════════════════════════════════════════════════
    st.header("🔢 **FASE 2: Vettorizzazione TF-IDF**")
    
    X = vectorizer.transform([clean])
    
    # Estraggo solo i token presenti (non-zero)
    nz_indices = X.nonzero()[1]
    present_tokens = [(vectorizer.get_feature_names_out()[i], float(X[0, i])) for i in nz_indices]
    present_tokens.sort(key=lambda x: x[1], reverse=True)  # Ordina per peso TF-IDF
    
    st.subheader("📊 Token estratti e loro pesi TF-IDF")
    
    if present_tokens:
        tfidf_df = pd.DataFrame(present_tokens, columns=["Token", "Peso TF-IDF"])
        tfidf_df["Peso TF-IDF"] = tfidf_df["Peso TF-IDF"].round(4)
        st.dataframe(tfidf_df, use_container_width=True)
        
        # Spiegazione TF-IDF
        with st.expander("🔍 Come funziona TF-IDF?"):
            st.markdown("""
            **TF-IDF** (Term Frequency-Inverse Document Frequency) calcola l'importanza di ogni parola:
            
            $$\\text{TF-IDF}(t,d) = \\text{TF}(t,d) \\times \\text{IDF}(t)$$
            
            Dove:
            - **TF(t,d)**: frequenza del termine t nel documento d
            - **IDF(t)**: $\\log\\frac{N}{df(t)}$ dove N = totale documenti, df(t) = documenti contenenti t
            
            **Risultato**: parole frequenti nel documento MA rare nel corpus ottengono peso maggiore.
            """)
            
            # Mostra alcuni calcoli di esempio
            st.markdown("**Esempio di calcolo per i primi 3 token:**")
            for i, (token, tfidf_score) in enumerate(present_tokens[:3]):
                vocab_idx = vectorizer.vocabulary_.get(token)
                st.write(f"• **{token}**: TF-IDF = {tfidf_score:.4f} (indice vocabolario: {vocab_idx})")
    else:
        st.warning("⚠️ Nessun token valido trovato dopo la pulizia!")
        st.stop()

    # ═══════════════════════════════════════════════════════════
    # SEZIONE 3: CALCOLO NAIVE BAYES DETTAGLIATO
    # ═══════════════════════════════════════════════════════════
    st.header("🧮 **FASE 3: Calcolo Naive Bayes (Passo per Passo)**")
    
    # Estraggo solo i nomi dei token per il calcolo
    token_names = [tok for tok, _ in present_tokens]
    
    st.subheader("🎯 Sentiment Analysis - Breakdown Completo")
    
    # Preparo la tabella step-by-step
    breakdown_rows = []
    log_scores = {}  # Per tenere traccia dei punteggi finali
    
    for cls_idx, cls_name in labels_sent.items():
        # 1. LOG PRIOR
        log_prior = clf_sent.class_log_prior_[cls_idx]
        log_scores[cls_idx] = log_prior
        
        breakdown_rows.append({
            "👥 Classe": cls_name,
            "📈 Passaggio": "🏁 Log Prior P(C)",
            "🔢 Valore": f"{log_prior:.4f}",
            "📊 Cumulativo": f"{log_prior:.4f}",
            "💡 Spiegazione": f"Frequenza di '{cls_name}' nel training set"
        })
        
        # 2. LOG LIKELIHOOD per ogni token
        for token in token_names:
            vocab_idx = vectorizer.vocabulary_.get(token)
            if vocab_idx is not None:
                log_likelihood = clf_sent.feature_log_prob_[cls_idx][vocab_idx]
                log_scores[cls_idx] += log_likelihood
                
                breakdown_rows.append({
                    "👥 Classe": "",
                    "📈 Passaggio": f"📝 + log P('{token}' | {cls_name})",
                    "🔢 Valore": f"{log_likelihood:.4f}",
                    "📊 Cumulativo": f"{log_scores[cls_idx]:.4f}",
                    "💡 Spiegazione": f"Quanto '{token}' è tipico di {cls_name}"
                })
    
    # Mostro la tabella completa
    breakdown_df = pd.DataFrame(breakdown_rows)
    st.dataframe(breakdown_df, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════
    # SEZIONE 4: CALCOLO PROBABILITÀ FINALI
    # ═══════════════════════════════════════════════════════════
    st.subheader("🎲 Conversione in Probabilità (Softmax)")
    
    log_neg = log_scores[0]
    log_pos = log_scores[1]
    
    # Calcolo softmax manuale
    exp_neg = np.exp(log_neg)
    exp_pos = np.exp(log_pos)
    prob_neg_manual = exp_neg / (exp_neg + exp_pos)
    prob_pos_manual = exp_pos / (exp_neg + exp_pos)
    
    st.markdown("**🧮 Applicazione della funzione Softmax:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.latex(f"P(\\text{{Negative}}|x) = \\frac{{e^{{{log_neg:.2f}}}}}{{e^{{{log_neg:.2f}}} + e^{{{log_pos:.2f}}}}} = {prob_neg_manual:.3f}")
    with col2:
        st.latex(f"P(\\text{{Positive}}|x) = \\frac{{e^{{{log_pos:.2f}}}}}{{e^{{{log_neg:.2f}}} + e^{{{log_pos:.2f}}}}} = {prob_pos_manual:.3f}")
    
    # Decisione manuale
    predicted_class_manual = "Positive 😀" if prob_pos_manual > prob_neg_manual else "Negative 😞"
    confidence_manual = max(prob_pos_manual, prob_neg_manual)
    
    st.success(f"🎯 **Predizione manuale**: {predicted_class_manual} (confidenza: {confidence_manual:.1%})")
    
    # ═══════════════════════════════════════════════════════════
    # SEZIONE 5: VERIFICA CON MODELLO SKLEARN
    # ═══════════════════════════════════════════════════════════
    st.subheader("✅ Verifica con sklearn")
    
    # Predizione automatica del modello
    probs_sent = clf_sent.predict_proba(X)[0]
    pred_sent = clf_sent.predict(X)[0]
    
    # Confronto
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🤖 Sklearn - Prob Negative", f"{probs_sent[0]:.3f}")
        st.metric("🧮 Calcolo Manuale", f"{prob_neg_manual:.3f}")
        diff_neg = abs(probs_sent[0] - prob_neg_manual)
        st.metric("📊 Differenza", f"{diff_neg:.6f}" if diff_neg > 0.0001 else "✅ Identico")
    
    with col2:
        st.metric("🤖 Sklearn - Prob Positive", f"{probs_sent[1]:.3f}")
        st.metric("🧮 Calcolo Manuale", f"{prob_pos_manual:.3f}")
        diff_pos = abs(probs_sent[1] - prob_pos_manual)
        st.metric("📊 Differenza", f"{diff_pos:.6f}" if diff_pos > 0.0001 else "✅ Identico")
    
    if diff_neg < 0.0001 and diff_pos < 0.0001:
        st.success("🎉 **PERFETTO!** I calcoli manuali corrispondono esattamente al modello sklearn!")
    else:
        st.warning("⚠️ Piccole differenze dovute agli arrotondamenti numerici")

    # ═══════════════════════════════════════════════════════════
    # SEZIONE 6: WATCH AGAIN PREDICTION
    # ═══════════════════════════════════════════════════════════
    st.header("🎬 **FASE 4: Predizione Watch Again**")
    
    probs_watch = clf_watch.predict_proba(X)[0]
    pred_watch = clf_watch.predict(X)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🚫 No Rewatch", f"{probs_watch[0]:.1%}")
    with col2:
        st.metric("🔄 Would Rewatch", f"{probs_watch[1]:.1%}")
    
    st.info(f"🎭 **Predizione Watch Again**: {labels_watch[pred_watch]}")

    # ═══════════════════════════════════════════════════════════
    # SEZIONE 7: RISULTATI FINALI E VISUALIZZAZIONI
    # ═══════════════════════════════════════════════════════════
    st.header("📊 **RISULTATI FINALI**")
    
    # Metriche riassuntive
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("😀 Sentiment", "Positive" if pred_sent == 1 else "Negative", 
                 f"{max(probs_sent):.0%}")
    with col2:
        st.metric("🎬 Rewatch", "Yes" if pred_watch == 1 else "No", 
                 f"{max(probs_watch):.0%}")
    with col3:
        st.metric("🎯 Confidenza Sentiment", f"{max(probs_sent):.1%}")
    with col4:
        st.metric("🎯 Confidenza Rewatch", f"{max(probs_watch):.1%}")
    
    # Grafici delle probabilità
    st.subheader("📈 Visualizzazione Probabilità")
    
    # Grafico Sentiment
    sent_df = pd.DataFrame({
        "Classe": ["Negative 😞", "Positive 😀"],
        "Probabilità": [probs_sent[0], probs_sent[1]]
    })
    
    chart_sent = alt.Chart(sent_df).mark_bar(size=50).encode(
        x=alt.X("Classe:N", title="Sentiment"),
        y=alt.Y("Probabilità:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilità"),
        color=alt.Color("Classe:N", scale=alt.Scale(range=["#ff6b6b", "#4ecdc4"])),
        tooltip=["Classe:N", alt.Tooltip("Probabilità:Q", format=".1%")]
    ).properties(width=300, height=200, title="Sentiment Analysis")
    
    # Grafico Watch Again
    watch_df = pd.DataFrame({
        "Classe": ["No Rewatch 🙁", "Would Rewatch 😊"],
        "Probabilità": [probs_watch[0], probs_watch[1]]
    })
    
    chart_watch = alt.Chart(watch_df).mark_bar(size=50).encode(
        x=alt.X("Classe:N", title="Watch Again"),
        y=alt.Y("Probabilità:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilità"),
        color=alt.Color("Classe:N", scale=alt.Scale(range=["#ffa726", "#66bb6a"])),
        tooltip=["Classe:N", alt.Tooltip("Probabilità:Q", format=".1%")]
    ).properties(width=300, height=200, title="Watch Again Prediction")
    
    # Mostro i grafici affiancati
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_sent, use_container_width=True)
    with col2:
        st.altair_chart(chart_watch, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# SEZIONI AGGIUNTIVE: ANALISI APPROFONDITE
# ═══════════════════════════════════════════════════════════

# —————— PARAMETRI INTERNI DEL MODELLO ——————
# Sostituisci la sezione "PARAMETRI INTERNI DEL MODELLO" con questo codice:

# —————— PARAMETRI INTERNI DEL MODELLO ——————
with st.expander("⚙️ **Parametri Interni del Modello Naive Bayes**"):
    st.markdown("### 🔧 Struttura del Modello")
    
    n_classes, n_features = clf_sent.feature_log_prob_.shape
    st.markdown(f"- **Numero classi**: {n_classes}")
    st.markdown(f"- **Dimensione vocabolario**: {n_features:,}")
    st.markdown(f"- **Smoothing parameter (α)**: {clf_sent.alpha}")
    
    # Analisi del token più pertinente (quello con TF-IDF più alto)
    if 'present_tokens' in locals() and present_tokens:
        st.markdown("### 🎯 Analisi del Token Più Pertinente")
        
        # Prendo il token con peso TF-IDF più alto
        most_relevant_token, highest_tfidf = present_tokens[0]  # Già ordinati per peso decrescente
        
        st.info(f"🔍 **Token analizzato**: '{most_relevant_token}' (TF-IDF: {highest_tfidf:.4f})")
        st.markdown("*Questo è il token con maggiore peso TF-IDF nella recensione, quindi il più discriminante.*")
        
        vocab_idx = vectorizer.vocabulary_.get(most_relevant_token)
        
        # Conteggi raw dal training set
        count_neg = clf_sent.feature_count_[0, vocab_idx]
        count_pos = clf_sent.feature_count_[1, vocab_idx]
        total_neg = clf_sent.class_count_[0]
        total_pos = clf_sent.class_count_[1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"📊 '{most_relevant_token}' in Negative", int(count_neg))
            st.metric("📚 Totale parole Negative", int(total_neg))
        with col2:
            st.metric(f"📊 '{most_relevant_token}' in Positive", int(count_pos))
            st.metric("📚 Totale parole Positive", int(total_pos))
        
        # Calcolo manuale delle probabilità con smoothing
        alpha = clf_sent.alpha
        V = n_features
        
        prob_neg = (count_neg + alpha) / (total_neg + alpha * V)
        prob_pos = (count_pos + alpha) / (total_pos + alpha * V)
        
        st.markdown("### 📐 Formula di Laplace Smoothing")
        st.latex(r"P(w|c) = \frac{\text{count}(w,c) + \alpha}{\sum_{w'} \text{count}(w',c) + \alpha \cdot V}")
        
        st.markdown(f"**Calcoli per '{most_relevant_token}':**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Classe Negative:**")
            st.latex(f"P({most_relevant_token}|\\text{{Neg}}) = \\frac{{{int(count_neg)} + {alpha}}}{{{int(total_neg)} + {alpha} \\times {V}}}")
            st.latex(f"= \\frac{{{int(count_neg + alpha)}}}{{{int(total_neg + alpha * V)}}} = {prob_neg:.6f}")
        
        with col2:
            st.markdown("**Classe Positive:**")
            st.latex(f"P({most_relevant_token}|\\text{{Pos}}) = \\frac{{{int(count_pos)} + {alpha}}}{{{int(total_pos)} + {alpha} \\times {V}}}")
            st.latex(f"= \\frac{{{int(count_pos + alpha)}}}{{{int(total_pos + alpha * V)}}} = {prob_pos:.6f}")
        
        # Verifica con i parametri memorizzati nel modello
        stored_log_prob_neg = clf_sent.feature_log_prob_[0, vocab_idx]
        stored_log_prob_pos = clf_sent.feature_log_prob_[1, vocab_idx]
        
        st.markdown("### ✅ Verifica con Parametri Memorizzati")
        
        verification_df = pd.DataFrame({
            "Classe": ["Negative", "Positive"],
            "Log Prob (Calcolato)": [f"{np.log(prob_neg):.6f}", f"{np.log(prob_pos):.6f}"],
            "Log Prob (Memorizzato)": [f"{stored_log_prob_neg:.6f}", f"{stored_log_prob_pos:.6f}"],
            "Match": ["✅" if abs(np.log(prob_neg) - stored_log_prob_neg) < 1e-10 else "❌",
                     "✅" if abs(np.log(prob_pos) - stored_log_prob_pos) < 1e-10 else "❌"]
        })
        
        st.dataframe(verification_df, use_container_width=True, hide_index=True)
        
        # Interpretazione del token
        st.markdown("### 🧠 Interpretazione")
        
        if prob_pos > prob_neg:
            ratio = prob_pos / prob_neg
            st.success(f"🎯 '{most_relevant_token}' è **{ratio:.1f}x più probabile** nelle recensioni positive")
            st.write(f"Questo token contribuisce **positivamente** alla classificazione della recensione.")
        else:
            ratio = prob_neg / prob_pos
            st.error(f"🎯 '{most_relevant_token}' è **{ratio:.1f}x più probabile** nelle recensioni negative")
            st.write(f"Questo token contribuisce **negativamente** alla classificazione della recensione.")
        
        # Contributo al punteggio finale
        log_contribution_neg = stored_log_prob_neg
        log_contribution_pos = stored_log_prob_pos
        
        st.markdown(f"**Contributo al punteggio finale:**")
        st.markdown(f"- Classe Negative: `{log_contribution_neg:.4f}`")
        st.markdown(f"- Classe Positive: `{log_contribution_pos:.4f}`")
        
        net_contribution = log_contribution_pos - log_contribution_neg
        if net_contribution > 0:
            st.success(f"📈 **Contributo netto**: +{net_contribution:.4f} verso Positive")
        else:
            st.error(f"📉 **Contributo netto**: {net_contribution:.4f} verso Negative")

# —————— TEORIA E SPIEGAZIONI ——————
with st.expander("📚 **Teoria del Naive Bayes**"):
    st.markdown("""
    ### 🎯 Teorema di Bayes
    
    Il classificatore Naive Bayes si basa sul **Teorema di Bayes**:
    
    $$P(C|D) = \\frac{P(D|C) \\cdot P(C)}{P(D)}$$
    
    Dove:
    - **P(C|D)**: probabilità della classe C dato il documento D (**posteriore**)
    - **P(D|C)**: probabilità del documento D data la classe C (**likelihood**)
    - **P(C)**: probabilità della classe C (**prior**)
    - **P(D)**: probabilità del documento D (**evidenza**)
    
    ### 🤝 Assunzione di Indipendenza
    
    Il termine "Naive" deriva dall'assunzione che le parole siano **condizionalmente indipendenti**:
    
    $$P(w_1, w_2, ..., w_n | C) = P(w_1|C) \\times P(w_2|C) \\times ... \\times P(w_n|C)$$
    
    ### 📊 In Log-Spazio
    
    Per evitare underflow numerico, lavoriamo in log-spazio:
    
    $$\\log P(C|D) = \\log P(C) + \\sum_{i=1}^{n} \\log P(w_i|C)$$
    
    ### ✅ Vantaggi
    - **Veloce**: training e predizione molto rapidi
    - **Robusto**: funziona bene anche con pochi dati
    - **Interpretabile**: facile capire quali parole influenzano la predizione
    - **Baseline efficace**: ottimi risultati per molti task di classificazione testo
    
    ### ⚠️ Svantaggi
    - **Assunzione irrealistica**: le parole non sono realmente indipendenti
    - **Sensibile ai dati di training**: performance dipende dalla qualità del dataset
    - **Problemi con token rari**: può dare troppo peso a parole molto rare
    """)

# —————— METRICHE GLOBALI DEL MODELLO ——————
st.header("📊 **Valutazione Globale del Modello**")

# Carico e processo tutto il dataset per le metriche
df_full = pd.read_csv("Synthetic Reviews.csv")
if "watch_again" not in df_full.columns:
    np.random.seed(42)
    df_full["watch_again"] = df_full["label"].apply(
        lambda l: int(np.random.rand() < (0.8 if l==1 else 0.1))
    )

df_full["clean"] = df_full["text"].apply(clean_text)
X_full = vectorizer.transform(df_full["clean"])

# Metriche Sentiment
y_true_sent = df_full["label"]
y_pred_sent = clf_sent.predict(X_full)
y_prob_sent = clf_sent.predict_proba(X_full)[:, 1]

acc_sent = accuracy_score(y_true_sent, y_pred_sent)
auc_sent = roc_auc_score(y_true_sent, y_prob_sent)
cm_sent = confusion_matrix(y_true_sent, y_pred_sent)

# Metriche Watch Again
y_true_watch = df_full["watch_again"]
y_pred_watch = clf_watch.predict(X_full)
y_prob_watch = clf_watch.predict_proba(X_full)[:, 1]

acc_watch = accuracy_score(y_true_watch, y_pred_watch)
auc_watch = roc_auc_score(y_true_watch, y_prob_watch)
cm_watch = confusion_matrix(y_true_watch, y_pred_watch)

# Visualizzazione metriche
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Accuracy Sentiment", f"{acc_sent:.1%}")
with col2:
    st.metric("📈 ROC AUC Sentiment", f"{auc_sent:.3f}")
with col3:
    st.metric("🎯 Accuracy Watch Again", f"{acc_watch:.1%}")
with col4:
    st.metric("📈 ROC AUC Watch Again", f"{auc_watch:.3f}")

# Confusion matrices
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔥 Confusion Matrix - Sentiment")
    cm_sent_df = pd.DataFrame(
        cm_sent,
        index=["Vero Negative", "Vero Positive"],
        columns=["Pred Negative", "Pred Positive"]
    )
    st.dataframe(cm_sent_df, use_container_width=True)

with col2:
    st.subheader("🎬 Confusion Matrix - Watch Again")
    cm_watch_df = pd.DataFrame(
        cm_watch,
        index=["Vero No", "Vero Yes"],
        columns=["Pred No", "Pred Yes"]
    )
    st.dataframe(cm_watch_df, use_container_width=True)

# —————— CURVA ROC ——————
st.subheader("📈 Curva ROC - Watch Again")

fpr, tpr, thresholds = roc_curve(y_true_watch, y_prob_watch)
roc_auc_manual = auc(fpr, tpr)

# Creo DataFrame per Altair
roc_df = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr,
    "Threshold": thresholds
})

# AGGIUNGI QUESTA PARTE CHE MANCAVA - Definizione del grafico ROC principale
roc_chart = alt.Chart(roc_df).mark_line(
    color='blue', strokeWidth=3
).encode(
    x=alt.X("False Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("True Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
    tooltip=["False Positive Rate:Q", "True Positive Rate:Q", "Threshold:Q"]
).properties(
    width=400, 
    height=400, 
    title=f"ROC Curve - Watch Again (AUC = {roc_auc_manual:.3f})"
)

# Linea diagonale (random classifier)
diagonal_df = pd.DataFrame({
    "False Positive Rate": [0, 1],
    "True Positive Rate": [0, 1]
})

diagonal_chart = alt.Chart(diagonal_df).mark_line(
    color='red', strokeWidth=2, strokeDash=[3, 3]
).encode(
    x="False Positive Rate:Q",
    y="True Positive Rate:Q"
)

# Combino i grafici
combined_roc = alt.layer(roc_chart, diagonal_chart).resolve_scale(
    color='independent'
)

st.altair_chart(combined_roc, use_container_width=True)

st.caption(f"📊 **AUC = {roc_auc_manual:.3f}** - Più alto è meglio (max = 1.0)")

# ═══════════════════════════════════════════════════════════
# AGGIUNTA PER SPIEGARE LA COSTRUZIONE DELLA CURVA ROC
# ═══════════════════════════════════════════════════════════

# Aggiungi questa sezione DOPO la visualizzazione della curva ROC esistente

# —————— SPIEGAZIONE DETTAGLIATA CURVA ROC ——————
with st.expander("🔬 **Come viene costruita la Curva ROC - Spiegazione Numerica**"):
    st.markdown("### 🎯 Processo di Costruzione della Curva ROC")
    
    st.markdown("""
    La **Curva ROC** (Receiver Operating Characteristic) visualizza le performance di un classificatore 
    binario al variare della soglia di decisione. Vediamo come viene costruita passo per passo.
    """)
    
    # Spiegazione teorica
    st.markdown("### 📐 Teoria: TPR e FPR")
    st.markdown("""
    Per ogni soglia di decisione, calcoliamo:
    - **TPR (True Positive Rate)** = Sensibilità = TP / (TP + FN)
    - **FPR (False Positive Rate)** = 1 - Specificità = FP / (FP + TN)
    
    Dove:
    - **TP** = Veri Positivi (predizione corretta per classe positiva)
    - **FN** = Falsi Negativi (predizione sbagliata per classe positiva)  
    - **FP** = Falsi Positivi (predizione sbagliata per classe negativa)
    - **TN** = Veri Negativi (predizione corretta per classe negativa)
    """)
    
    # Calcolo dettagliato con alcune soglie di esempio
    st.markdown("### 🔢 Calcolo Numerico per Soglie Specifiche")
    
    # Prendo un subset dei dati per rendere più chiaro l'esempio
    sample_size = min(100, len(y_true_watch))  # Limito a 100 esempi per chiarezza
    y_true_sample = y_true_watch[:sample_size]
    y_prob_sample = y_prob_watch[:sample_size]
    
    # Definisco alcune soglie di esempio
    example_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    calculation_results = []
    
    for threshold in example_thresholds:
        # Predizioni binarie basate sulla soglia
        y_pred_threshold = (y_prob_sample >= threshold).astype(int)
        
        # Calcolo matrice di confusione
        tn = np.sum((y_true_sample == 0) & (y_pred_threshold == 0))
        fp = np.sum((y_true_sample == 0) & (y_pred_threshold == 1))
        fn = np.sum((y_true_sample == 1) & (y_pred_threshold == 0))
        tp = np.sum((y_true_sample == 1) & (y_pred_threshold == 1))
        
        # Calcolo TPR e FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        calculation_results.append({
            "Soglia": f"{threshold:.1f}",
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "TPR": f"{tpr:.3f}",
            "FPR": f"{fpr:.3f}"
        })
    
    # Mostro la tabella dei calcoli
    calc_df = pd.DataFrame(calculation_results)
    st.dataframe(calc_df, use_container_width=True, hide_index=True)
    
    st.markdown("### 📊 Interpretazione della Tabella")
    st.markdown("""
    - **Soglia alta (0.9)**: Poche predizioni positive → Basso FPR ma anche basso TPR
    - **Soglia bassa (0.1)**: Molte predizioni positive → Alto TPR ma anche alto FPR
    - **Soglia media (0.5)**: Bilanciamento tra TPR e FPR
    """)
    
    # Visualizzazione del processo completo
    st.markdown("### 🎨 Costruzione della Curva Completa")
    
    # Ricalcolo con tutte le soglie per l'intero dataset
    fpr_full, tpr_full, thresholds_full = roc_curve(y_true_watch, y_prob_watch)
    
    st.markdown(f"""
    Il processo completo per il nostro dataset:
    1. **Ordiniamo** tutte le probabilità predette: {len(np.unique(y_prob_watch))} valori unici
    2. **Testiamo** {len(thresholds_full)} soglie diverse
    3. **Calcoliamo** TPR e FPR per ogni soglia
    4. **Plottiamo** i punti (FPR, TPR) per creare la curva
    """)
    
    # Mostra alcuni punti della curva
    st.markdown("### 📈 Punti Significativi della Curva")
    
    # Selezioni alcuni punti interessanti della curva
    n_points = min(10, len(fpr_full))
    indices = np.linspace(0, len(fpr_full)-1, n_points, dtype=int)
    
    curve_points = pd.DataFrame({
        "Punto": [f"#{i+1}" for i in range(n_points)],
        "Soglia": [f"{thresholds_full[idx]:.3f}" for idx in indices],
        "FPR": [f"{fpr_full[idx]:.3f}" for idx in indices],
        "TPR": [f"{tpr_full[idx]:.3f}" for idx in indices],
        "Coordinate": [f"({fpr_full[idx]:.3f}, {tpr_full[idx]:.3f})" for idx in indices]
    })
    
    st.dataframe(curve_points, use_container_width=True, hide_index=True)
    
    # Spiegazione dell'AUC
    st.markdown("### 🏆 Calcolo dell'AUC (Area Under Curve)")
    
    # Calcolo manuale dell'AUC usando il metodo dei trapezi
    manual_auc = np.trapz(tpr_full, fpr_full)
    sklearn_auc = auc(fpr_full, tpr_full)
    
    st.markdown(f"""
    L'**AUC** rappresenta l'area sotto la curva ROC:
    
    - **Calcolo manuale** (regola dei trapezi): {manual_auc:.6f}
    - **Calcolo sklearn**: {sklearn_auc:.6f}
    - **Differenza**: {abs(manual_auc - sklearn_auc):.8f}
    
    **Interpretazione AUC**:
    - **1.0**: Classificatore perfetto
    - **0.5**: Classificatore casuale (linea diagonale)
    - **< 0.5**: Peggio del caso (ma può essere invertito)
    """)
    
    # Formula matematica per AUC
    st.markdown("### 🧮 Formula Matematica")
    st.latex(r"AUC = \int_0^1 TPR(FPR^{-1}(t)) \, dt")
    st.markdown("Approssimata numericamente con la regola dei trapezi:")
    st.latex(r"AUC \approx \sum_{i=0}^{n-1} \frac{(FPR_{i+1} - FPR_i) \cdot (TPR_{i+1} + TPR_i)}{2}")
    
    # Mostra il calcolo dei primi trapezi
    st.markdown("### 🔺 Calcolo dei Primi Trapezi (Esempio)")
    
    trapezoid_examples = []
    for i in range(min(5, len(fpr_full)-1)):
        base = fpr_full[i+1] - fpr_full[i]
        height_avg = (tpr_full[i+1] + tpr_full[i]) / 2
        area = base * height_avg
        
        trapezoid_examples.append({
            "Trapezio": f"#{i+1}",
            "Base": f"{base:.6f}",
            "Altezza Media": f"{height_avg:.6f}",
            "Area": f"{area:.8f}"
        })
    
    trap_df = pd.DataFrame(trapezoid_examples)
    st.dataframe(trap_df, use_container_width=True, hide_index=True)
    
    total_example_area = sum(float(row["Area"]) for row in trapezoid_examples)
    st.info(f"📊 Somma delle prime {len(trapezoid_examples)} aree: {total_example_area:.8f}")
    
    # Confronto con classificatore casuale
    st.markdown("### 🎲 Confronto con Classificatore Casuale")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🤖 Nostro Modello", f"AUC = {sklearn_auc:.3f}")
        if sklearn_auc > 0.8:
            st.success("🎯 Excellent!")
        elif sklearn_auc > 0.7:
            st.info("👍 Good")
        elif sklearn_auc > 0.6:
            st.warning("⚠️ Fair")
        else:
            st.error("❌ Poor")
    
    with col2:
        st.metric("🎲 Classificatore Casuale", "AUC = 0.500")
        improvement = (sklearn_auc - 0.5) / 0.5 * 100
        st.metric("📈 Miglioramento", f"+{improvement:.1f}%")
    
    # Punti critici della curva
    st.markdown("### 🎯 Punti Critici della Curva")
    
    # Trova il punto ottimale (massima distanza dalla diagonale)
    optimal_idx = np.argmax(tpr_full - fpr_full)
    optimal_threshold = thresholds_full[optimal_idx]
    optimal_tpr = tpr_full[optimal_idx]
    optimal_fpr = fpr_full[optimal_idx]
    
    st.markdown(f"""
    **Punto Ottimale** (massima distanza dalla diagonale):
    - **Soglia**: {optimal_threshold:.3f}
    - **Coordinate**: ({optimal_fpr:.3f}, {optimal_tpr:.3f})
    - **Distanza dalla diagonale**: {(optimal_tpr - optimal_fpr):.3f}
    """)
    
    # Visualizzazione interattiva della soglia ottimale
    st.markdown("### 🎚️ Visualizzazione Soglia Ottimale")
    
    # Creo DataFrame per visualizzare il punto ottimale
    optimal_point_df = pd.DataFrame({
        "False Positive Rate": [optimal_fpr],
        "True Positive Rate": [optimal_tpr],
        "Type": ["Optimal Point"]
    })
    
    # Grafico con punto ottimale evidenziato
    base_roc = alt.Chart(roc_df).mark_line(color='blue', strokeWidth=2).encode(
        x=alt.X("False Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("True Positive Rate:Q", scale=alt.Scale(domain=[0, 1]))
    )
    
    optimal_point = alt.Chart(optimal_point_df).mark_circle(
        size=200, color='red', stroke='white', strokeWidth=2
    ).encode(
        x="False Positive Rate:Q",
        y="True Positive Rate:Q",
        tooltip=[
            alt.Tooltip("False Positive Rate:Q", format=".3f"),
            alt.Tooltip("True Positive Rate:Q", format=".3f"),
            "Type:N"
        ]
    )
    
    combined_chart = (base_roc + optimal_point).properties(
        width=400, height=300, title="ROC Curve con Punto Ottimale"
    )
    
    st.altair_chart(combined_chart, use_container_width=True)
    
    # Spiegazione finale
    st.markdown("### 🎓 Riassunto")
    st.markdown(f"""
    La curva ROC del nostro modello mostra:
    
    1. **AUC = {sklearn_auc:.3f}** → Prestazioni {"eccellenti" if sklearn_auc > 0.8 else "buone" if sklearn_auc > 0.7 else "discrete"}
    2. **Punto ottimale** alla soglia {optimal_threshold:.3f}
    3. **Curva ben distanziata** dalla diagonale casuale
    4. **Costruita testando {len(thresholds_full)} soglie** diverse
    
    Questo conferma che il nostro modello Naive Bayes ha imparato a distinguere 
    efficacemente tra recensioni che porterebbero a un rewatch e quelle che non lo farebbero.
    """)

# ═══════════════════════════════════════════════════════════
# SEZIONE AGGIUNTIVA: THRESHOLD ANALYSIS INTERATTIVA
# ═══════════════════════════════════════════════════════════

st.header("🎚️ **Analisi Interattiva delle Soglie**")

st.markdown("""
Esplora come cambia la performance del modello al variare della soglia di decisione.
Questa sezione ti permette di vedere in tempo reale l'effetto delle diverse soglie sui risultati.
""")

# Slider per la soglia
threshold_slider = st.slider(
    "🎯 Seleziona la soglia di decisione:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Soglia per convertire probabilità in predizioni binarie"
)

# Calcolo metriche per la soglia selezionata
y_pred_threshold = (y_prob_watch >= threshold_slider).astype(int)

# Matrice di confusione per la soglia selezionata
tn_t = np.sum((y_true_watch == 0) & (y_pred_threshold == 0))
fp_t = np.sum((y_true_watch == 0) & (y_pred_threshold == 1))
fn_t = np.sum((y_true_watch == 1) & (y_pred_threshold == 0))
tp_t = np.sum((y_true_watch == 1) & (y_pred_threshold == 1))

# Calcolo metriche
precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
f1_t = 2 * (precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
accuracy_t = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
specificity_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0

# Visualizzazione metriche
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Accuracy", f"{accuracy_t:.3f}")
with col2:
    st.metric("🔍 Precision", f"{precision_t:.3f}")
with col3:
    st.metric("📈 Recall (TPR)", f"{recall_t:.3f}")
with col4:
    st.metric("🎪 F1-Score", f"{f1_t:.3f}")

# Matrice di confusione per la soglia selezionata
st.subheader(f"🔢 Matrice di Confusione (Soglia = {threshold_slider:.2f})")

col1, col2 = st.columns(2)
with col1:
    confusion_matrix_threshold = pd.DataFrame({
        "Pred: No Rewatch": [tn_t, fn_t],
        "Pred: Rewatch": [fp_t, tp_t]
    }, index=["True: No Rewatch", "True: Rewatch"])
    
    st.dataframe(confusion_matrix_threshold, use_container_width=True)

with col2:
    st.markdown("**Interpretazione:**")
    st.markdown(f"- **Veri Positivi (TP)**: {tp_t} - Correttamente predetto 'Rewatch'")
    st.markdown(f"- **Veri Negativi (TN)**: {tn_t} - Correttamente predetto 'No Rewatch'")
    st.markdown(f"- **Falsi Positivi (FP)**: {fp_t} - Erroneamente predetto 'Rewatch'")
    st.markdown(f"- **Falsi Negativi (FN)**: {fn_t} - Erroneamente predetto 'No Rewatch'")

# Posizione sulla curva ROC
st.subheader("📍 Posizione sulla Curva ROC")

current_fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
current_tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0

st.info(f"🎯 Punto attuale sulla curva ROC: ({current_fpr:.3f}, {current_tpr:.3f})")

# Trova il punto più vicino sulla curva ROC calcolata
distances = np.sqrt((fpr_full - current_fpr)**2 + (tpr_full - current_tpr)**2)
closest_idx = np.argmin(distances)
closest_threshold = thresholds_full[closest_idx]

st.markdown(f"🔍 Soglia più vicina nella curva ROC: {closest_threshold:.3f}")

# Visualizzazione del punto sulla curva
current_point_df = pd.DataFrame({
    "False Positive Rate": [current_fpr],
    "True Positive Rate": [current_tpr],
    "Type": [f"Soglia {threshold_slider:.2f}"]
})

roc_with_point = alt.Chart(roc_df).mark_line(color='blue', strokeWidth=2).encode(
    x=alt.X("False Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("True Positive Rate:Q", scale=alt.Scale(domain=[0, 1]))
)

current_point_chart = alt.Chart(current_point_df).mark_circle(
    size=300, color='orange', stroke='black', strokeWidth=2
).encode(
    x="False Positive Rate:Q",
    y="True Positive Rate:Q",
    tooltip=[
        alt.Tooltip("False Positive Rate:Q", format=".3f"),
        alt.Tooltip("True Positive Rate:Q", format=".3f"),
        "Type:N"
    ]
)

# Linea diagonale
diagonal_line = alt.Chart(diagonal_df).mark_line(
    color='red', strokeWidth=1, strokeDash=[3, 3]
).encode(
    x="False Positive Rate:Q",
    y="True Positive Rate:Q"
)

interactive_roc = (roc_with_point + current_point_chart + diagonal_line).properties(
    width=500, height=400, title=f"Curva ROC con Soglia {threshold_slider:.2f}"
)

st.altair_chart(interactive_roc, use_container_width=True)

# Analisi del trade-off
st.subheader("⚖️ Analisi del Trade-off")

if current_tpr > current_fpr:
    trade_off_quality = "Buono"
    trade_off_color = "green"
else:
    trade_off_quality = "Scarso"
    trade_off_color = "red"

st.markdown(f"""
**Trade-off TPR vs FPR:**
- **TPR (Sensibilità)**: {current_tpr:.3f} - % di veri positivi identificati correttamente
- **FPR (1-Specificità)**: {current_fpr:.3f} - % di veri negativi classificati erroneamente come positivi
- **Trade-off**: {trade_off_quality} (TPR {'>' if current_tpr > current_fpr else '<='} FPR)
""")

# Raccomandazione sulla soglia
if threshold_slider < 0.3:
    st.warning("⚠️ **Soglia Bassa**: Molte predizioni positive, alta recall ma bassa precision")
elif threshold_slider > 0.7:
    st.warning("⚠️ **Soglia Alta**: Poche predizioni positive, alta precision ma bassa recall")
else:
    st.success("✅ **Soglia Bilanciata**: Buon compromesso tra precision e recall")

# Distribuzione delle probabilità
st.subheader("📊 Distribuzione delle Probabilità Predette")

prob_df = pd.DataFrame({
    "Probabilità": y_prob_watch,
    "Classe Vera": ["Rewatch" if y == 1 else "No Rewatch" for y in y_true_watch]
})

prob_hist = alt.Chart(prob_df).mark_bar(opacity=0.7).encode(
    x=alt.X("Probabilità:Q", bin=alt.Bin(maxbins=30)),
    y=alt.Y("count():Q"),
    color=alt.Color("Classe Vera:N", scale=alt.Scale(range=["#ff6b6b", "#4ecdc4"])),
    tooltip=["count():Q", "Classe Vera:N"]
).properties(width=600, height=300, title="Distribuzione delle Probabilità per Classe")

# Linea verticale per la soglia corrente
threshold_line = alt.Chart(pd.DataFrame({"Soglia": [threshold_slider]})).mark_rule(
    color='red', strokeWidth=3, strokeDash=[5, 5]
).encode(
    x="Soglia:Q"
)

prob_chart = (prob_hist + threshold_line).resolve_scale(color='independent')
st.altair_chart(prob_chart, use_container_width=True)

st.caption(f"📍 La linea rossa indica la soglia corrente ({threshold_slider:.2f}). Le predizioni a destra della linea sono classificate come 'Rewatch'.")

# —————— TOP FEATURES ANALYSIS ——————
st.header("🔍 **Analisi Top Features**")

st.subheader("🎯 Token più Discriminanti per Sentiment")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**😞 Top Negative Words**")
    negative_df = pd.DataFrame({
        "Token": ts[:10],
        "Peso": ps[:10]
    })
    st.dataframe(negative_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**😀 Top Positive Words**")
    positive_df = pd.DataFrame({
        "Token": ts[-10:],
        "Peso": ps[-10:]
    })
    st.dataframe(positive_df, use_container_width=True, hide_index=True)

st.subheader("🎬 Token più Discriminanti per Watch Again")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🙁 Top No-Rewatch Words**")
    no_rewatch_df = pd.DataFrame({
        "Token": tw[:10],
        "Peso": pw[:10]
    })
    st.dataframe(no_rewatch_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**😊 Top Rewatch Words**")
    rewatch_df = pd.DataFrame({
        "Token": tw[-10:],
        "Peso": pw[-10:]
    })
    st.dataframe(rewatch_df, use_container_width=True, hide_index=True)

# —————— DISTRIBUZIONE LUNGHEZZA RECENSIONI ——————
st.header("📏 **Analisi Distribuzione Dataset**")

# Calcolo lunghezza recensioni
df_full["text_length"] = df_full["text"].str.len()
df_full["word_count"] = df_full["text"].str.split().str.len()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribuzione Lunghezza Caratteri")
    
    length_chart = alt.Chart(df_full).mark_bar(
        opacity=0.7, binSpacing=2
    ).encode(
        x=alt.X("text_length:Q", bin=alt.Bin(maxbins=30), title="Lunghezza (caratteri)"),
        y=alt.Y("count():Q", title="Frequenza"),
        color=alt.Color("label:N", 
                       scale=alt.Scale(range=["#ff6b6b", "#4ecdc4"]),
                       legend=alt.Legend(title="Sentiment")),
        tooltip=["count():Q"]
    ).properties(width=300, height=200)
    
    st.altair_chart(length_chart, use_container_width=True)

with col2:
    st.subheader("📝 Distribuzione Numero Parole")
    
    words_chart = alt.Chart(df_full).mark_bar(
        opacity=0.7, binSpacing=2
    ).encode(
        x=alt.X("word_count:Q", bin=alt.Bin(maxbins=25), title="Numero parole"),
        y=alt.Y("count():Q", title="Frequenza"),
        color=alt.Color("watch_again:N", 
                       scale=alt.Scale(range=["#ffa726", "#66bb6a"]),
                       legend=alt.Legend(title="Watch Again")),
        tooltip=["count():Q"]
    ).properties(width=300, height=200)
    
    st.altair_chart(words_chart, use_container_width=True)

# Statistiche descrittive
st.subheader("📈 Statistiche Descrittive")

stats_df = pd.DataFrame({
    "Metrica": ["Lunghezza media (caratteri)", "Lunghezza mediana", "Parole medie", "Parole mediane"],
    "Valore": [
        f"{df_full['text_length'].mean():.1f}",
        f"{df_full['text_length'].median():.1f}",
        f"{df_full['word_count'].mean():.1f}",
        f"{df_full['word_count'].median():.1f}"
    ]
})

st.dataframe(stats_df, use_container_width=True, hide_index=True)

# —————— ANALISI CORRELAZIONE ——————
st.header("🔗 **Analisi delle Correlazioni**")

# Correlazione tra sentiment e watch_again
correlation = df_full[["label", "watch_again", "text_length", "word_count"]].corr()

st.subheader("📊 Matrice di Correlazione")

# Creo heatmap con Altair
corr_data = []
for i, row_name in enumerate(correlation.index):
    for j, col_name in enumerate(correlation.columns):
        corr_data.append({
            "Variable 1": row_name,
            "Variable 2": col_name,
            "Correlation": correlation.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_data)

heatmap = alt.Chart(corr_df).mark_rect().encode(
    x=alt.X("Variable 1:O", title=""),
    y=alt.Y("Variable 2:O", title=""),
    color=alt.Color("Correlation:Q", 
                   scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                   legend=alt.Legend(title="Correlazione")),
    tooltip=["Variable 1:O", "Variable 2:O", alt.Tooltip("Correlation:Q", format=".3f")]
).properties(width=300, height=300)

# Aggiungo testo con i valori
text = alt.Chart(corr_df).mark_text(
    fontSize=12, fontWeight="bold"
).encode(
    x="Variable 1:O",
    y="Variable 2:O",
    text=alt.Text("Correlation:Q", format=".2f"),
    color=alt.condition(
        alt.datum.Correlation > 0.5,
        alt.value("white"),
        alt.value("black")
    )
)

st.altair_chart(heatmap + text, use_container_width=True)

# Interpretazione correlazioni
st.markdown("### 🎯 Interpretazione delle Correlazioni")

sentiment_watch_corr = correlation.loc["label", "watch_again"]
sentiment_length_corr = correlation.loc["label", "text_length"]

if sentiment_watch_corr > 0.3:
    st.success(f"✅ **Correlazione Sentiment-Rewatch**: {sentiment_watch_corr:.3f} - Correlazione positiva forte!")
    st.write("Le recensioni positive tendono ad essere associate a maggiore propensione al rewatch.")
elif sentiment_watch_corr > 0.1:
    st.info(f"📊 **Correlazione Sentiment-Rewatch**: {sentiment_watch_corr:.3f} - Correlazione positiva moderata.")
else:
    st.warning(f"⚠️ **Correlazione Sentiment-Rewatch**: {sentiment_watch_corr:.3f} - Correlazione debole.")

if abs(sentiment_length_corr) > 0.2:
    direction = "positive" if sentiment_length_corr > 0 else "negative"
    st.info(f"📏 **Correlazione Sentiment-Lunghezza**: {sentiment_length_corr:.3f} - Le recensioni più lunghe tendono ad essere più {direction}.")


# Informazioni finali
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    🎓 <strong>Progetto Accademico</strong> - AI e ML per il Marketing - IULM<br>
    👨‍🎓 Luca Tallarico (1034109) - Anno 2024/2025<br>
    🤖 Powered by Streamlit + scikit-learn + Altair
</div>
""", unsafe_allow_html=True)