import { useState } from "react";
import axios from "axios";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [url, setUrl] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [loadingMsg, setLoadingMsg] = useState("");

  const messages = [
    "Fetching reviews from Amazon...",
    "Running reviews through AI model...",
    "Ranking by helpfulness...",
    "Almost done..."
  ];

  const analyze = async () => {
    if (!url.includes("amazon")) {
      setError("Please paste a valid Amazon product URL.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);

    // Cycle through loading messages
    let i = 0;
    setLoadingMsg(messages[0]);
    const interval = setInterval(() => {
      i = (i + 1) % messages.length;
      setLoadingMsg(messages[i]);
    }, 2000);

    try {
      const res = await axios.post(`${API}/scrape-and-analyze`, { url });
      setResults(res.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Something went wrong. Is the backend running?"
      );
    }

    clearInterval(interval);
    setLoading(false);
  };

  return (
    <div style={styles.page}>
      <div style={styles.panel}>

        {/* Header */}
        <div style={styles.header}>
          <span style={styles.icon}>🧹</span>
          <div>
            <h1 style={styles.title}>ReviewSweeperAI</h1>
            <p style={styles.subtitle}>Find reviews worth reading — instantly</p>
          </div>
        </div>

        {/* URL Input */}
        <div style={styles.inputWrap}>
          <input
            style={styles.input}
            placeholder="Paste Amazon product URL here..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && analyze()}
          />
          <button
            onClick={analyze}
            style={styles.btn}
            disabled={loading || !url}
          >
            {loading ? "..." : "Sweep"}
          </button>
        </div>

        {error && <p style={styles.error}>{error}</p>}

        {/* Loading State */}
        {loading && (
          <div style={styles.loadingBox}>
            <div style={styles.spinner} />
            <p style={styles.loadingMsg}>{loadingMsg}</p>
          </div>
        )}

        {/* Results */}
        {results && !loading && (
          <div>
            <div style={styles.statsRow}>
              <span>📊 {results.total_analyzed} reviews analyzed</span>
              <span>✅ {results.helpful_count} helpful</span>
              <span style={styles.productName}>{results.product_title || ""}</span>
            </div>

            <Section
              title="🟢 Most Helpful Positive"
              reviews={results.top_positive}
              color="#22c55e"
              empty="No helpful positive reviews found."
            />
            <Section
              title="🔴 Most Helpful Critical"
              reviews={results.top_critical}
              color="#ef4444"
              empty="No helpful critical reviews found."
            />
          </div>
        )}

      </div>
    </div>
  );
}

function Section({ title, reviews, color, empty }) {
  return (
    <div style={{ marginBottom: "1.25rem" }}>
      <h2 style={{ color, fontSize: "0.95rem", fontWeight: 700, marginBottom: "0.5rem" }}>
        {title}
      </h2>
      {reviews.length === 0
        ? <p style={{ color: "#64748b", fontSize: "0.85rem", fontStyle: "italic" }}>{empty}</p>
        : reviews.map((r, i) => <ReviewCard key={i} review={r} color={color} />)
      }
    </div>
  );
}

function ReviewCard({ review, color }) {
  return (
    <div style={{ ...styles.card, borderLeft: `3px solid ${color}` }}>
      {review.title && (
        <p style={styles.cardTitle}>{review.title}</p>
      )}
      <p style={styles.cardText}>{review.text}</p>
      <div style={styles.cardMeta}>
        <span>{"⭐".repeat(Math.min(review.rating || 0, 5))}</span>
        <span style={styles.badge}>
          {Math.round(review.confidence * 100)}% helpful
        </span>
        {review.url && (
          <a href={review.url} target="_blank" rel="noreferrer" style={styles.link}>
            View on Amazon →
          </a>
        )}
      </div>
    </div>
  );
}

const styles = {
  page: { minHeight: "100vh", background: "#0f172a", display: "flex", justifyContent: "center", padding: "2rem 1rem", fontFamily: "system-ui, sans-serif" },
  panel: { width: "100%", maxWidth: 480, color: "#e2e8f0" },
  header: { display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.5rem" },
  icon: { fontSize: "2rem" },
  title: { margin: 0, fontSize: "1.4rem", fontWeight: 800, color: "#38bdf8" },
  subtitle: { margin: 0, fontSize: "0.85rem", color: "#64748b" },
  inputWrap: { display: "flex", gap: "0.5rem", marginBottom: "0.75rem" },
  input: { flex: 1, padding: "0.6rem 0.75rem", borderRadius: 8, border: "1px solid #334155", background: "#1e293b", color: "#e2e8f0", fontSize: "0.9rem", outline: "none" },
  btn: { padding: "0.6rem 1.1rem", borderRadius: 8, border: "none", background: "#38bdf8", color: "#0f172a", fontWeight: 700, cursor: "pointer", fontSize: "0.95rem", whiteSpace: "nowrap" },
  error: { color: "#ef4444", fontSize: "0.85rem", margin: "0.5rem 0" },
  loadingBox: { display: "flex", flexDirection: "column", alignItems: "center", padding: "2rem", gap: "0.75rem" },
  spinner: { width: 28, height: 28, border: "3px solid #334155", borderTop: "3px solid #38bdf8", borderRadius: "50%", animation: "spin 0.8s linear infinite" },
  loadingMsg: { color: "#64748b", fontSize: "0.9rem", margin: 0 },
  statsRow: { display: "flex", gap: "1rem", fontSize: "0.8rem", color: "#64748b", marginBottom: "1rem", flexWrap: "wrap" },
  productName: { color: "#94a3b8", fontStyle: "italic" },
  card: { background: "#1e293b", borderRadius: 8, padding: "0.75rem", marginBottom: "0.6rem", border: "1px solid #1e293b" },
  cardTitle: { margin: "0 0 0.3rem", fontWeight: 600, fontSize: "0.9rem", color: "#e2e8f0" },
  cardText: { margin: "0 0 0.5rem", fontSize: "0.85rem", color: "#94a3b8", lineHeight: 1.5 },
  cardMeta: { display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" },
  badge: { background: "#0f172a", padding: "2px 8px", borderRadius: 10, fontSize: "0.75rem", color: "#38bdf8" },
  link: { color: "#38bdf8", fontSize: "0.8rem", textDecoration: "none" },
};