"""
Review Sentiment Analyser — SVM (LinearSVC) Backend
Run: python app.py
Visit: http://localhost:5000

Folder structure expected:
  project/
  ├── app.py
  ├── svm_best_model.pkl
  └── templates/
      └── index.html

Install dependencies:
  pip install flask scikit-learn joblib
"""

import re
import string
import joblib
import os
from flask import Flask, request, jsonify, render_template

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
STOP_WORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than','too',
    'very','s','t','can','will','just','don','should','now','d','ll','m','o',
    're','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven',
    'isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won',
    'wouldn'
])


def preprocess(text: str) -> str:
    """Mirror the notebook's NLP preprocessing pipeline."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'svm_best_model.pkl')
model = joblib.load(MODEL_PATH)
print(f"✅  Model loaded: {MODEL_PATH}")

# ── FLASK APP ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data        = request.get_json(force=True)
        text        = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided.'}), 400

        clean       = preprocess(text)
        token_count = len(clean.split())
        prediction  = int(model.predict([clean])[0])
        decision    = float(model.decision_function([clean])[0])

        return jsonify({
            'prediction':     prediction,
            'label':          'Positive' if prediction == 1 else 'Negative',
            'decision_score': decision,
            'token_count':    token_count,
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 SentimentIQ running...\n")

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )