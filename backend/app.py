from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os
import joblib

app = Flask(__name__)

# ── CORS - Allow local dev + all Vercel deployments ─────────────────────────
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "https://*.vercel.app"
])

# ── Load your 3 trained files once at startup ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler        = joblib.load(os.path.join(BASE_DIR, 'models/scaler_1.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'models/label_encoder_1.pkl'))
model         = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models/dass_cnn_model_1.keras'))

# ── Anxiety level mapping (1-5 numeric → text) ─────────────────────────────
ANXIETY_LABELS = {
    1: "Normal",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
    5: "Extremely Severe"
}

# ── DASS-21 question metadata ───────────────────────────────────────────────
QUESTIONS = [
    (1,  "I found it hard to wind down",                                        "S"),
    (2,  "I was aware of dryness of my mouth",                                  "A"),
    (3,  "I couldn't seem to experience any positive feeling",                  "D"),
    (4,  "I experienced breathing difficulty",                                  "A"),
    (5,  "I found it difficult to work up the initiative to do things",         "D"),
    (6,  "I tended to over-react to situations",                                "S"),
    (7,  "I experienced trembling (e.g. in the hands)",                         "A"),
    (8,  "I felt that I was using a lot of nervous energy",                     "S"),
    (9,  "I was worried about situations in which I might panic",               "A"),
    (10, "I felt that I had nothing to look forward to",                        "D"),
    (11, "I found myself getting agitated",                                     "S"),
    (12, "I found it difficult to relax",                                       "S"),
    (13, "I felt down-hearted and blue",                                        "D"),
    (14, "I was intolerant of anything that kept me from getting on",           "S"),
    (15, "I felt I was close to panic",                                         "A"),
    (16, "I was unable to become enthusiastic about anything",                  "D"),
    (17, "I felt I wasn't worth much as a person",                              "D"),
    (18, "I felt that I was rather touchy",                                     "S"),
    (19, "I was aware of the action of my heart in the absence of physical exertion", "A"),
    (20, "I felt scared without any good reason",                               "A"),
    (21, "I felt that life was meaningless",                                    "D"),
]

def severity_label(score, subscale):
    """Return severity string based on DASS-21 clinical thresholds."""
    thresholds = {
        "D": [(0,9,"Normal"),(10,13,"Mild"),(14,20,"Moderate"),(21,27,"Severe"),(28,42,"Extremely Severe")],
        "A": [(0,7,"Normal"),(8,9,"Mild"),(10,14,"Moderate"),(15,19,"Severe"),(20,42,"Extremely Severe")],
        "S": [(0,14,"Normal"),(15,18,"Mild"),(19,25,"Moderate"),(26,33,"Severe"),(34,42,"Extremely Severe")],
    }
    for lo, hi, label in thresholds[subscale]:
        if lo <= score <= hi:
            return label
    return "Unknown"

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        answers = data.get('answers', [])

        if len(answers) != 21:
            return jsonify({'error': 'Exactly 21 answers required'}), 400

        for i, a in enumerate(answers):
            if a not in [0, 1, 2, 3]:
                return jsonify({'error': f'Answer {i+1} must be 0-3'}), 400

        # ── Preprocess ───────────────────────────────────────────────────
        answers = [int(a) for a in answers]
        X        = np.array(answers, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_cnn    = X_scaled.reshape(1, 21, 1)

        # ── Predict ──────────────────────────────────────────────────────
        probs     = model.predict(X_cnn, verbose=0)[0]
        pred_idx  = int(np.argmax(probs))
        pred_num  = int(label_encoder.inverse_transform([pred_idx])[0])
        pred_label = ANXIETY_LABELS.get(pred_num, str(pred_num))
        confidence = float(np.max(probs))

        # ── All class probabilities ───────────────────────────────────────
        all_probs = {}
        for i, p in enumerate(probs):
            num = int(label_encoder.inverse_transform([i])[0])
            label = ANXIETY_LABELS.get(num, str(num))
            all_probs[label] = round(float(p) * 100, 1)

        # ── DASS-21 subscale scoring ──────────────────────────────────────
        d_items = [3, 5, 10, 13, 16, 17, 21]
        a_items = [2, 4, 7, 9, 15, 19, 20]
        s_items = [1, 6, 8, 11, 12, 14, 18]

        d_score = sum(answers[i-1] for i in d_items) * 2
        a_score = sum(answers[i-1] for i in a_items) * 2
        s_score = sum(answers[i-1] for i in s_items) * 2

        subscales = {
            "Depression": {"score": d_score, "severity": severity_label(d_score, "D")},
            "Anxiety":    {"score": a_score, "severity": severity_label(a_score, "A")},
            "Stress":     {"score": s_score, "severity": severity_label(s_score, "S")},
        }

        return jsonify({
            'prediction': pred_label,
            'confidence': round(confidence * 100, 1),
            'all_probs':  all_probs,
            'subscales':  subscales,
            'd_score':    d_score,
            'a_score':    a_score,
            's_score':    s_score,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
