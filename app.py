import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

@st.cache_resource
def load_model():
    # expects models/spam_model.joblib in repo
    return joblib.load("models/spam_model.joblib")

def predict_and_score(model, text: str):
    proba = float(model.predict_proba([text])[0, 1])
    pred  = int(proba >= 0.5)
    return ("🛑 SPAM" if pred else "✅ HAM"), proba

def explain_message(model, text, top_k=10):
    """Show which n-grams push toward SPAM vs HAM for this one message."""
    pipe = model
    tfidf = pipe.named_steps["tfidf"]
    clf   = pipe.named_steps["clf"]

    X = tfidf.transform([text])           # 1 x V sparse vector
    coefs = clf.coef_[0]                  # shape (V,)
    X = X.tocoo()

    feats = tfidf.get_feature_names_out()
    contrib = {}
    for i, v in zip(X.col, X.data):
        contrib[feats[i]] = v * coefs[i]  # approximate contribution

    if not contrib:
        return [], []

    items = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    top_spam = [(k, round(v, 4)) for k, v in items[:top_k]]
    top_ham  = [(k, round(v, 4)) for k, v in items[-top_k:]][::-1]
    return top_spam, top_ham

st.title("📩 SMS Spam Classifier")
st.write("Paste a message and the model will predict whether it's **Spam** or **Ham**.")

model = load_model()

text = st.text_area("Message", height=150, placeholder="e.g., Congratulations! You've won a prize...")

col1, col2 = st.columns([1,1])
with col1:
    thresh = st.slider("Decision threshold (spam)", 0.10, 0.90, 0.50, 0.05)
with col2:
    explain = st.checkbox("Show top contributing n-grams", value=True)

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        proba = float(model.predict_proba([text])[0, 1])
        pred  = proba >= thresh
        st.markdown(f"**Prediction:** {'🛑 SPAM' if pred else '✅ HAM'}")
        st.write(f"Spam score: `{proba:.3f}`  •  Threshold: `{thresh:.2f}`")

        if explain:
            top_spam, top_ham = explain_message(model, text, top_k=8)
            st.markdown("#### Why it thinks so (approximate):")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Pushes **toward SPAM**")
                if top_spam:
                    st.table(top_spam)
                else:
                    st.write("—")
            with c2:
                st.caption("Pushes **toward HAM**")
                if top_ham:
                    st.table(top_ham)
                else:
                    st.write("—")

st.markdown("---")
st.caption("Model: TF-IDF (1–2 grams) + Logistic Regression • Stored in `models/spam_model.joblib`")
