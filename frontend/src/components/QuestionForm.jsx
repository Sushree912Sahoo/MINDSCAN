import { useState } from "react";

const SECTIONS = [
  {
    id: "stress",
    label: "Stress Scale",
    labelClass: "section-label stress",
    icon: "🔥",
    iconClass: "si-s",
    subTitle: "Stress Indicators",
    subDesc: "Persistent arousal, tension, agitation",
    selClass: "sel-stress",
    questions: [
      { id: 1,  text: "I found it hard to wind down" },
      { id: 6,  text: "I tended to over-react to situations" },
      { id: 8,  text: "I felt that I was using a lot of nervous energy" },
      { id: 11, text: "I found myself getting agitated" },
      { id: 12, text: "I found it difficult to relax" },
      { id: 14, text: "I was intolerant of anything that kept me from getting on with what I was doing" },
      { id: 18, text: "I felt that I was rather touchy" },
    ],
  },
  {
    id: "anxiety",
    label: "Anxiety Scale",
    labelClass: "section-label anxiety",
    icon: "⚡",
    iconClass: "si-a",
    subTitle: "Anxiety Indicators",
    subDesc: "Autonomic arousal, fear, panic responses",
    selClass: "sel-anxiety",
    questions: [
      { id: 2,  text: "I was aware of dryness of my mouth" },
      { id: 4,  text: "I experienced breathing difficulty (e.g. rapid breathing, breathlessness)" },
      { id: 7,  text: "I experienced trembling (e.g. in the hands)" },
      { id: 9,  text: "I was worried about situations in which I might panic and make a fool of myself" },
      { id: 15, text: "I felt I was close to panic" },
      { id: 19, text: "I was aware of the action of my heart in the absence of physical exertion" },
      { id: 20, text: "I felt scared without any good reason" },
    ],
  },
  {
    id: "depression",
    label: "Depression Scale",
    labelClass: "section-label depression",
    icon: "🌧️",
    iconClass: "si-d",
    subTitle: "Depression Indicators",
    subDesc: "Low affect, hopelessness, anhedonia",
    selClass: "sel-depression",
    questions: [
      { id: 3,  text: "I couldn't seem to experience any positive feeling at all" },
      { id: 5,  text: "I found it difficult to work up the initiative to do things" },
      { id: 10, text: "I felt that I had nothing to look forward to" },
      { id: 13, text: "I felt down-hearted and blue" },
      { id: 16, text: "I was unable to become enthusiastic about anything" },
      { id: 17, text: "I felt I wasn't worth much as a person" },
      { id: 21, text: "I felt that life was meaningless" },
    ],
  },
];

const OPTIONS = [
  { value: 0, label: "Never" },
  { value: 1, label: "Sometimes" },
  { value: 2, label: "Often" },
  { value: 3, label: "Almost Always" },
];

export default function QuestionForm({ onSubmit, loading, error }) {
  const [answers, setAnswers] = useState({});

  const totalAnswered  = Object.keys(answers).length;
  const totalQuestions = 21;
  const progress       = Math.round((totalAnswered / totalQuestions) * 100);

  const handleSelect = (qid, value) => {
    setAnswers((prev) => ({ ...prev, [qid]: value }));
  };

  const handleSubmit = () => {
    if (totalAnswered < totalQuestions) {
      alert(`Please answer all 21 questions. ${totalQuestions - totalAnswered} remaining.`);
      return;
    }
    const ordered = Array.from({ length: 21 }, (_, i) => answers[i + 1] ?? 0);
    onSubmit(ordered);
  };

  return (
    <div>
      {/* Hero */}
      <div className="hero">
        <div className="hero-badge">DASS-21 Assessment Tool</div>
        <h1>Mental Health<br/><em>Self-Assessment</em></h1>
        <p>
          Rate how much each statement applied to you over the <strong>past week</strong>.
          There are no right or wrong answers.
        </p>
        <div className="hero-chips">
          <span className="chip">21 Questions</span>
          <span className="chip">~4 minutes</span>
          <span className="chip">CNN Powered</span>
          <span className="chip">Anonymous</span>
        </div>
      </div>

      {/* Instructions */}
      <div className="instructions">
        <strong>Rate each statement from 0 to 3:</strong>
        <div className="scale-row">
          {OPTIONS.map((o) => (
            <div className="scale-item" key={o.value}>
              <div className={`scale-dot sd${o.value}`}>{o.value}</div>
              <span style={{ fontSize: "12px", color: "var(--text-lt)" }}>{o.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Progress */}
      <div className="progress-wrap">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <span className="progress-label">{totalAnswered} / {totalQuestions} answered</span>
      </div>

      {/* Sections */}
      {SECTIONS.map((section) => (
        <div key={section.id}>
          {/* Section header */}
          <div className="section-header">
            <div className={`section-icon ${section.iconClass}`}>{section.icon}</div>
            <span className={section.labelClass}>{section.label}</span>
            <span className="section-count">
              {section.questions.filter((q) => answers[q.id] !== undefined).length}
              /{section.questions.length}
            </span>
          </div>

          {/* Subscale block */}
          <div className="subscale-block">
            <div className="subscale-header">
              <div className={`section-icon ${section.iconClass}`}>{section.icon}</div>
              <div>
                <h3>{section.subTitle}</h3>
                <p>{section.subDesc}</p>
              </div>
            </div>

            {section.questions.map((q, idx) => (
              <div
                className="question-card"
                key={q.id}
                style={{ animationDelay: `${idx * 0.04}s` }}
              >
                <div className="q-top">
                  <span className="q-num">Q{q.id}.</span>
                  <span className="q-text">{q.text}</span>
                </div>
                <div className="options">
                  {OPTIONS.map((opt) => {
                    const isSelected = answers[q.id] === opt.value;
                    return (
                      <button
                        key={opt.value}
                        className={`option-btn ${isSelected ? section.selClass : ""}`}
                        onClick={() => handleSelect(q.id, opt.value)}
                        type="button"
                      >
                        <span className="opt-score">{opt.value}</span>
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      {error && <div className="error-box">⚠ {error}</div>}

      <button
        className="btn-primary"
        onClick={handleSubmit}
        disabled={loading || totalAnswered < totalQuestions}
      >
        {loading
          ? "Analysing with CNN model…"
          : totalAnswered < totalQuestions
          ? `Answer ${totalQuestions - totalAnswered} more question${totalQuestions - totalAnswered !== 1 ? "s" : ""} to continue`
          : "Generate My Assessment →"}
      </button>
    </div>
  );
}
