import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, request, jsonify
import numpy as np
import joblib
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler        = joblib.load(os.path.join(BASE_DIR, 'models/scaler_1.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'models/label_encoder_1.pkl'))

# Load TFLite model
interpreter = tflite.Interpreter(
    model_path=os.path.join(BASE_DIR, 'models/dass_model.tflite')
)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ANXIETY_LABELS = {
    1: "Normal",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
    5: "Extremely Severe"
}

def severity_label(score, subscale):
    thresholds = {
        "D": [(0,9,"Normal"),(10,13,"Mild"),(14,20,"Moderate"),(21,27,"Severe"),(28,42,"Extremely Severe")],
        "A": [(0,7,"Normal"),(8,9,"Mild"),(10,14,"Moderate"),(15,19,"Severe"),(20,42,"Extremely Severe")],
        "S": [(0,14,"Normal"),(15,18,"Mild"),(19,25,"Moderate"),(26,33,"Severe"),(34,42,"Extremely Severe")],
    }
    for lo, hi, label in thresholds[subscale]:
        if lo <= score <= hi:
            return label
    return "Unknown"

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        data    = request.get_json()
        answers = data.get('answers', [])

        if len(answers) != 21:
            return jsonify({'error': 'Exactly 21 answers required'}), 400

        for i, a in enumerate(answers):
            if a not in [0, 1, 2, 3]:
                return jsonify({'error': f'Answer {i+1} must be 0-3'}), 400

        answers  = [int(a) for a in answers]
        X        = np.array(answers, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_cnn    = X_scaled.reshape(1, 21, 1).astype(np.float32)

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], X_cnn)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx   = int(np.argmax(probs))
        pred_num   = int(label_encoder.inverse_transform([pred_idx])[0])
        pred_label = ANXIETY_LABELS.get(pred_num, str(pred_num))
        confidence = float(np.max(probs))

        all_probs = {}
        for i, p in enumerate(probs):
            num   = int(label_encoder.inverse_transform([i])[0])
            label = ANXIETY_LABELS.get(num, str(num))
            all_probs[label] = round(float(p) * 100, 1)

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
