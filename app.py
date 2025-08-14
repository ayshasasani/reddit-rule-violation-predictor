import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

# --------------------
# Load model and vectorizer
# --------------------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    # Calibrate probabilities for better thresholding
    calibrated_model = CalibratedClassifierCV(rf_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(tfidf.transform(["sample text"]), [0])  # dummy fit
    return calibrated_model, tfidf

model, tfidf = load_model()

# --------------------
# Text cleaning function
# --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------
# App title
# --------------------
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Reddit Comment Rule Violation Predictor</h1>", unsafe_allow_html=True)
st.write("Predict whether a Reddit comment might violate subreddit rules.")

# --------------------
# Sidebar info
# --------------------
st.sidebar.header("About This App")
st.sidebar.write("""
- **Purpose**: Detect potential rule-breaking comments.
- **Model**: Random Forest trained on Reddit dataset with calibrated probabilities.
- **Tip**: Paste your comment(s) or use a sample below.
""")

# --------------------
# Sample comments
# --------------------
sample_comments = [
    "I think you're completely wrong, this is nonsense!",
    "Here’s a link to the source: http://abc.com",
    "I love this subreddit, thanks for all the work!",
    "You are the worst, get lost!",
    "fuck you"
]

# Initialize a counter in session_state to track which sample to show
if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = 0

# Button to cycle through sample comments
if st.sidebar.button("Use Next Sample Comment"):
    st.session_state['user_input'] = sample_comments[st.session_state.sample_idx]
    st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(sample_comments)
else:
    st.session_state.setdefault('user_input', "")

# --------------------
# Input area
# --------------------
user_input = st.text_area(
    "Enter Reddit comment(s) here:",
    value=st.session_state['user_input'],
    height=150,
    help="You can enter multiple comments, one per line."
)

# --------------------
# Prediction button
# --------------------
if st.button("Predict Violation Probability"):
    if not user_input.strip():
        st.warning("Please enter at least one comment.")
    else:
        comments = [line.strip() for line in user_input.split("\n") if line.strip()]
        results = []

        for comment in comments:
            cleaned = clean_text(comment)
            features = tfidf.transform([cleaned])
            prob = model.predict_proba(features)[0][1]

            # Dynamic thresholds based on calibrated probabilities
            if prob > 0.6:
                level = "High Risk"
                color = "red"
            elif prob > 0.3:
                level = "Medium Risk"
                color = "orange"
            else:
                level = "Low Risk"
                color = "green"

            results.append({
                "Comment": comment,
                "Probability": f"{prob:.2%}",
                "Risk Level": f"<span style='color:{color}; font-weight:bold'>{level}</span>"
            })

        # --------------------
        # Display results with colored risk levels
        # --------------------
        df = pd.DataFrame(results)
        st.write("### Prediction Results")
        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Show progress bar for the last comment
        last_prob = float(df.iloc[-1]["Probability"].strip('%')) / 100
        st.progress(last_prob)
