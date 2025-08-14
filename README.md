Reddit Rule Violation Predictor
A Streamlit web app that predicts whether a Reddit comment may violate subreddit rules using a trained Random Forest model.

Features
Enter single or multiple Reddit comments for prediction

Sample comments provided for testing

Displays violation probability and color-coded risk level (Low, Medium, High)

Quick, real-time predictions

Installation
Clone the repository:
git clone https://github.com/ayshasasani/reddit-rule-violation-predictor.git
cd reddit-rule-violation-predictor

Create and activate a virtual environment:
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt
Ensure app.py, model.pkl, and tfidf.pkl are in the folder.

Running the App
streamlit run app.py
Model Details
Model: Random Forest Classifier

Features: TF-IDF on combined comment text, rule, and examples

Evaluation: Accuracy ~62%

Future Improvements
Use transformer-based models (BERT) for better contextual understanding

Add database integration to store predictions

Support batch comment moderation and multi-language detection

References
Kaggle: Jigsaw Agile Community Rules Competition
Vaswani, A., et al. (2017). Attention is All You Need
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
