const SEV_CLASS = {
  "Normal":           "sev-normal",
  "Mild":             "sev-mild",
  "Moderate":         "sev-moderate",
  "Severe":           "sev-severe",
  "Extremely Severe": "sev-extremely",
};

const CARDS = [
  { key: "Stress",     cls: "stress",     icon: "🔥" },
  { key: "Anxiety",    cls: "anxiety",    icon: "⚡" },
  { key: "Depression", cls: "depression", icon: "🌧️" },
];

function summaryMessage(subscales) {
  const entries = Object.entries(subscales);
  const allNormal = entries.every(([, d]) => d.severity === "Normal");

  if (allNormal)
    return "Your scores across all three dimensions fall within the <strong>normal range</strong>. Keep maintaining your healthy routines and self-care practices.";

  const issues = entries
    .filter(([, d]) => d.severity !== "Normal")
    .map(([name, d]) => `your <strong>${name.toLowerCase()} level</strong> is ${d.severity.toLowerCase()}`);

  let msg = "The assessment indicates that " + issues.join(", ") + ". ";

  const highSeverity = entries.some(([, d]) =>
    d.severity === "Severe" || d.severity === "Extremely Severe"
  );
  if (highSeverity) {
    msg += "These scores suggest significant psychological distress. <strong>Please consider speaking with a mental health professional</strong> for proper evaluation and support.";
  } else {
    msg += "Mild to moderate scores often respond well to self-care strategies such as regular exercise, mindfulness, adequate sleep, and social support. Consider speaking with a counsellor if symptoms persist.";
  }
  return msg;
}

export default function ResultCard({ result, onReset }) {
  const { prediction, confidence, all_probs, subscales } = result;

  return (
    <div className="result-wrap">

      {/* Top hero */}
      <div className="result-hero">
        <p className="result-label">CNN Model Prediction</p>
        <h2 className="result-title">{prediction}</h2>
        <p className="result-conf">
          Model confidence: <strong>{confidence}%</strong>
        </p>
      </div>

      {/* Subscale cards */}
      <div className="subscale-grid">
        {CARDS.map(({ key, cls, icon }) => {
          const data = subscales[key];
          if (!data) return null;
          const sevCls = SEV_CLASS[data.severity] || "sev-normal";
          const barPct = Math.min((data.score / 42) * 100, 100);
          return (
            <div className={`subscale-card ${cls}`} key={key}>
              <span className="sub-icon">{icon}</span>
              <p className="sub-name">{key}</p>
              <p className="sub-score">
                {data.score}
                <span className="sub-score-max">/42</span>
              </p>
              <span className={`sub-severity ${sevCls}`}>
                <span className="sev-dot"></span>
                {data.severity}
              </span>
              <div className="score-bar-wrap">
                <div className="score-bar-fill" style={{ width: `${barPct}%` }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Probability distribution */}
      <div className="prob-card">
        <h3>📊 Class Probability Distribution</h3>
        {Object.entries(all_probs)
          .sort((a, b) => b[1] - a[1])
          .map(([label, pct]) => (
            <div className="prob-row" key={label}>
              <span className="prob-label">{label}</span>
              <div className="prob-bar-wrap">
                <div className="prob-bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <span className="prob-pct">{pct}%</span>
            </div>
          ))}
      </div>

      {/* Summary */}
      <div className="summary-note">
        <h3>📋 Interpretation Summary</h3>
        <p
          dangerouslySetInnerHTML={{
            __html: summaryMessage(subscales),
          }}
        />
        <div className="disclaimer">
          ⚠️ <strong>Disclaimer:</strong> This tool is for informational and
          screening purposes only. It does not constitute a clinical diagnosis.
          If you are experiencing significant distress, please consult a
          qualified mental health professional.
        </div>
      </div>

      {/* Reset */}
      <div className="result-actions">
        <button className="btn-secondary" onClick={onReset}>
          ↩ Retake Assessment
        </button>
      </div>

    </div>
  );
}
