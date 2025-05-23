import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 1) Carica dataset e artefatti
df = pd.read_csv("Synthetic Reviews.csv")
with open("vectorizer.pkl","rb") as f:   vectorizer = pickle.load(f)
with open("nb_model.pkl","rb") as f:      clf_sent   = pickle.load(f)
with open("watch_model.pkl","rb") as f:   clf_watch  = pickle.load(f)

# 2) Preprocessing
def clean_text(s):
    import re
    s = s.lower()
    return re.sub(r"[^a-z\s]","",s)
df["clean"] = df["text"].apply(clean_text)

# 3) Split (opzionale, se vuoi misurare su un test set)
from sklearn.model_selection import train_test_split
X_train, X_test, y_sent_train, y_sent_test = train_test_split(
    vectorizer.transform(df["clean"]), df["label"], test_size=0.2, random_state=42
)

# 4) Metriche Sentiment
p_sent = clf_sent.predict_proba(X_test)[:,1]
auc_sent = roc_auc_score(y_sent_test, p_sent)
mse_sent = mean_squared_error(y_sent_test, clf_sent.predict(X_test))

# 5) Metriche Watch-again
# (se hai creato df["watch_again"] prima nel notebook, altrimenti saltalo)
df["watch_again"] = df["label"].apply(lambda l: int(np.random.rand() < (0.8 if l==1 else 0.1)))
Xw = vectorizer.transform(df["clean"])
yw = df["watch_again"]
p_watch = clf_watch.predict_proba(Xw)[:,1]
auc_watch = roc_auc_score(yw, p_watch)
mse_watch = mean_squared_error(yw, clf_watch.predict(Xw))

# 6) Stampa metriche
print(f"Sentiment NB — AUC: {auc_sent:.3f},  MSE: {mse_sent:.3f}")
print(f"Watch-again NB — AUC: {auc_watch:.3f},  MSE: {mse_watch:.3f}")

plt.figure()
RocCurveDisplay.from_predictions(
    y_sent_test,
    p_sent,
    name="Sentiment"
).plot()

RocCurveDisplay.from_predictions(
    yw,
    p_watch,
    name="Watch Again"
).plot()

plt.title("ROC Curves")
plt.savefig("roc_curves.png")


# 8) Estrai top-10 token log-prob da nb
feat_names = vectorizer.get_feature_names_out()
logp_sent = clf_sent.feature_log_prob_[1]
logp_watch= clf_watch.feature_log_prob_[1]
top_idx_s  = np.argsort(logp_sent)[-10:][::-1]
top_idx_w  = np.argsort(logp_watch)[-10:][::-1]

top_tokens_s = feat_names[top_idx_s]
top_probs_s  = logp_sent[top_idx_s]
top_tokens_w = feat_names[top_idx_w]
top_probs_w  = logp_watch[top_idx_w]

# 9) Grafico top-token
fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].barh(top_tokens_s, top_probs_s); axes[0].set_title("Top‐10 token Positive")
axes[1].barh(top_tokens_w, top_probs_w); axes[1].set_title("Top‐10 token Watch Again")
fig.tight_layout()
plt.savefig("top_tokens.png")
