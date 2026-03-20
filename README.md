# MindScan — DASS-21 Anxiety Classification

**Live App:** https://mindscanweb.vercel.app/ 
**Backend API:** https://dass21-backend.onrender.com/api/health

A clinical web application that classifies **Anxiety Level** from the DASS-21
questionnaire using a trained 1D-CNN. Built with React + Flask, deployed on
Vercel (frontend) and Render (backend).

---

## What This Project Does

### The Problem
Mental health conditions — especially **anxiety, depression, and stress** — are
widely underdiagnosed. The DASS-21 (Depression Anxiety Stress Scales) is a
validated 21-item psychometric instrument used globally by clinicians to screen
for all three simultaneously. Manual scoring is slow and inaccessible to most
people.

### What the Model Predicts
The CNN was trained on the **Anxiety_Level** column of the DASS.csv dataset.
It performs **multi-class classification** over these severity levels:

| Predicted Class | Clinical Meaning |
|---|---|
| Normal | Scores within the healthy range |
| Mild | Slightly elevated — worth self-monitoring |
| Moderate | Clinically significant — counselling recommended |
| Severe | High distress — professional consultation strongly advised |
| Extremely Severe | Acute distress — immediate professional help recommended |

### DASS-21 Subscale Scoring (shown alongside CNN output)
In addition to the CNN's overall anxiety classification, the app independently
computes three clinical subscale scores per the official DASS-21 manual
(raw item sum × 2):

| Subscale | Columns Used | Score Range |
|---|---|---|
| Stress | Q3_1_S1 … Q3_7_S7 | 0–42 |
| Anxiety | Q3_8_A1 … Q3_14_A7 | 0–42 |
| Depression | Q3_15_D1 … Q3_21_D7 | 0–42 |

**Official thresholds (score × 2):**

| Level | Stress | Anxiety | Depression |
|---|---|---|---|
| Normal | 0–14 | 0–7 | 0–9 |
| Mild | 15–18 | 8–9 | 10–13 |
| Moderate | 19–25 | 10–14 | 14–20 |
| Severe | 26–33 | 15–19 | 21–27 |
| Extremely Severe | 34+ | 20+ | 28+ |

---

## Your CNN Architecture

Trained in Google Colab with TensorFlow/Keras. Saved with:
```python
model.save("dass_model.tflite")
joblib.dump(scaler, "scaler_1.pkl")
joblib.dump(le, "label_encoder_1.pkl")
```

### Model Summary

```
Input shape: (21, 1)
│
├─ Conv1D(16, kernel_size=3, activation='relu', padding='same')
├─ BatchNormalization()
│
├─ Conv1D(32, kernel_size=3, activation='relu', padding='same')
├─ BatchNormalization()
├─ MaxPooling1D(pool_size=2)
│
├─ Conv1D(32, kernel_size=3, activation='relu', padding='same')
├─ GlobalMaxPooling1D()
│
├─ Dense(128, activation='relu')
├─ Dropout(0.5)
│
└─ Dense(num_classes, activation='softmax')

Optimizer  : Adam(learning_rate=0.005)
Loss       : categorical_crossentropy
Callbacks  : EarlyStopping(patience=10) + ReduceLROnPlateau(patience=5)
Batch size : 32   |   Max epochs: 100   |   Val split: 20%
```

**Why 1D-CNN for DASS-21?**
The 21 DASS items are semantically ordered — Stress (S1–S7), Anxiety (A1–A7),
Depression (D1–D7). A 1D-CNN slides convolutional filters across adjacent
items to learn local co-occurrence patterns (e.g. "high S3 + high S5 = stress
overload") that a plain Dense/MLP would miss.

`BatchNormalization` after each Conv block stabilises activations and speeds up
convergence. `ReduceLROnPlateau` halves the learning rate when validation loss
plateaus, preventing oscillation near the optimum.

---

## End-to-End Prediction Pipeline

```
User answers 21 questions (values 0–3 each)
              │
              ▼
React builds ordered array in training column order:
  [Q3_1_S1, Q3_2_S2, ..., Q3_7_S7,      ← Stress   (idx 0–6)
   Q3_8_A1, Q3_9_A2, ..., Q3_14_A7,     ← Anxiety  (idx 7–13)
   Q3_15_D1, ..., Q3_21_D7]             ← Depression(idx 14–20)
              │
              ▼
POST /api/predict  →  Flask receives { "answers": [2, 0, 3, ...] }
              │
              ▼
scaler_1.pkl
  StandardScaler.transform(X)            shape: (1, 21)
  (same scaler fitted during training)
              │
              ▼
reshape → (1, 21, 1)                     adds channel dim for CNN
              │
              ▼
dass_model.tflite
  Conv1D(16) → BatchNorm                 detects basic item patterns
  Conv1D(32) → BatchNorm → MaxPool       deeper patterns, downsampled
  Conv1D(32) → GlobalMaxPool            strongest feature per map
  Dense(128) → Dropout(0.5)             combines all features
  Dense(n, softmax)                     class probabilities
              │
              ▼
label_encoder_1.pkl
  inverse_transform(argmax(probs))       index → "Moderate" etc.
              │
              ▼
DASS-21 subscale scoring
  Stress score     = sum(answers[0:7])  × 2
  Anxiety score    = sum(answers[7:14]) × 2
  Depression score = sum(answers[14:21])× 2
  Each mapped to severity label via clinical thresholds
              │
              ▼
Flask returns JSON:
  {
    "prediction": "Moderate",
    "confidence": 78.4,
    "all_probs":  { "Normal": 5.1, "Mild": 12.3, "Moderate": 78.4, ... },
    "subscales":  {
      "Stress":     { "score": 24, "severity": "Moderate" },
      "Anxiety":    { "score": 18, "severity": "Severe" },
      "Depression": { "score": 12, "severity": "Mild" }
    }
  }
              │
              ▼
React renders ResultCard:
  • Overall CNN class + confidence bar
  • Three subscale cards (Stress / Anxiety / Depression)
  • Softmax probability distribution chart
  • Clinical interpretation summary
```

---

## Trained Files — What Goes Where

| File | Saved as | Place in |
|---|---|---|
| CNN model | `dass_model.tflite` | `backend/models/` |
| StandardScaler | `scaler_1.pkl` | `backend/models/` |
| LabelEncoder | `label_encoder_1.pkl` | `backend/models/` |

```
dass21-app/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── models/
│       ├── dass_model.tflite        ← model.save("dass_model.tflite")
│       ├── scaler_1.pkl             ← joblib.dump(scaler, "scaler_1.pkl")
│       └── label_encoder_1.pkl      ← joblib.dump(le, "label_encoder_1.pkl")
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── main.jsx
│   │   └── components/
│   │       ├── Header.jsx
│   │       ├── QuestionForm.jsx
│   │       └── ResultCard.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── vercel.json
└── .gitignore
```

---

## Dataset & Research Basis

- **Instrument:** DASS-21 — Depression Anxiety Stress Scales (short form)
- **Original authors:** Lovibond & Lovibond, 1995, University of New South Wales
- **Mendeley Dataset:** https://data.mendeley.com/datasets/br82d4xkj7/1
- **Training target column:** `Anxiety_Level`
- **Input features:** 21 columns `Q3_1_S1` … `Q3_21_D7` (integer 0–3 each)

---

## STEP-BY-STEP SETUP

### STEP 1 — Install Prerequisites

| Tool | Version | Download |
|---|---|---|
| Python | 3.10 | https://www.python.org/downloads/ |
| Node.js | 18+ | https://nodejs.org/ |
| Git | latest | https://git-scm.com/ |
| VS Code | latest | https://code.visualstudio.com/ |

---

### STEP 2 — Place Your 3 Trained Files

Download from your Colab session and copy into `backend/models/`:

```
backend/models/dass_cnn_model_1.keras
backend/models/scaler_1.pkl
backend/models/label_encoder_1.pkl
```

> These are the exact filenames saved by your notebook.
> Do NOT rename them — `app.py` loads them by these exact names.

---

### STEP 3 — Run the Flask Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac / Linux)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Start Flask
python app.py
```

Flask starts at **http://localhost:5000**

Test the health check in your browser:
```
http://localhost:5000/api/health
→ { "classes": [...], "model_loaded": true, "n_features": 21, "status": "ok" }
```

---

### STEP 4 — Run the React Frontend

Open a **second terminal** in VS Code (keep Flask running in the first):

```bash
cd frontend
npm install
npm run dev
```

React starts at **http://localhost:5173**

---

### STEP 5 — Push to GitHub

```bash
# From the root dass21-app/ folder:
git init
git add .
git commit -m "Initial commit: DASS-21 CNN anxiety classification web app"

# Create repo at github.com/new → name it: dass21-mindscanner
git remote add origin https://github.com/YOUR_USERNAME/dass21-mindscanner.git
git branch -M main
git push -u origin main
```

---

### STEP 6 — Deploy Backend on Render (free)

> Vercel is not used for the backend — TensorFlow exceeds Vercel's 50 MB
> function size limit. Render handles large Python dependencies.

1. Go to **https://render.com** → sign in with GitHub
2. **New → Web Service** → connect `dass21-mindscanner`
3. Settings:
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Click **Create Web Service** → wait ~3 minutes
5. Copy your Render URL: `https://dass21-backend.onrender.com`

**Upload model files** (choose one option):

| Option | When to use | How |
|---|---|---|
| A — Push to git | Files < 100 MB total | Remove `*.keras` and `*.pkl` from `.gitignore`, then `git add . && git push` |
| B — Render Disk | Files > 100 MB | Render dashboard → your service → Disks → upload manually |

---

### STEP 7 — Deploy Frontend on Vercel

1. Go to **https://vercel.com** → sign in with GitHub
2. **Add New Project** → import `dass21-mindscanner`
3. Settings:
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. **Environment Variable:**
   - Key: `VITE_API_URL`
   - Value: `https://dass21-backend.onrender.com`
5. Click **Deploy**
6. Live URL: `https://dass21-mindscanner.vercel.app`

---

### STEP 8 — End-to-End Test

1. Open your Vercel URL
2. Answer all 21 DASS questions (0–3 for each)
3. Click **Generate My Assessment**
4. Results page shows:
   - **CNN prediction** (e.g. "Moderate") with confidence %
   - **Stress / Anxiety / Depression** cards with individual scores and severity badges
   - **Probability distribution** bar chart (softmax output)
   - **Clinical interpretation** summary

---

## Local Development Quick Reference

```bash
# Terminal 1 — Backend
cd backend && source venv/bin/activate && python app.py

# Terminal 2 — Frontend
cd frontend && npm run dev

# Push any changes
git add . && git commit -m "Update" && git push
```

Auto-redeploy times after `git push`:
- **Vercel** (frontend): ~1 minute
- **Render** (backend): ~2–3 minutes

---

## Clinical Disclaimer

This application is built for **research and informational purposes only**.
The DASS-21 is a screening tool, not a diagnostic instrument. Results from
this model should not replace professional clinical assessment. If you or
someone you know is experiencing significant psychological distress, please
consult a qualified mental health professional.
