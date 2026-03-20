import { useState } from "react";
import QuestionForm from "./components/QuestionForm";
import ResultCard from "./components/ResultCard";
import Header from "./components/Header";
import "./App.css";

export default function App() {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [step, setStep]       = useState("form");

  const handleSubmit = async (answers) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL || "http://localhost:5000"}/api/predict`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ answers }),
        }
      );
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
      setStep("result");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setStep("form");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="app">
      <Header />
      <main className="main">
        {step === "form" && (
          <QuestionForm onSubmit={handleSubmit} loading={loading} error={error} />
        )}
        {step === "result" && result && (
          <ResultCard result={result} onReset={handleReset} />
        )}
      </main>
      <footer className="footer">
        <p>This tool is for informational purposes only. Not a clinical diagnosis.</p>
        <a
          className="code-link"
          href="https://github.com/Sushree912Sahoo/MINDSCAN"
          target="_blank"
          rel="noopener noreferrer"
        >
          View Source Code
        </a>
      </footer>
    </div>
  );
}
