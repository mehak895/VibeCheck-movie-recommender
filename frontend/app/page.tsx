"use client";

import { useState } from "react";

export default function Home() {
  const [movie, setMovie] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const getRecommendations = async () => {
    if (!movie) return;

    setLoading(true);

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/recommend?movie=${encodeURIComponent(movie)}&user_id=3`
      );

      const data = await res.json();

      // ✅ SAFE HANDLING (prevents crash)
      if (!Array.isArray(data)) {
        console.log("Unexpected response:", data);
        setResults([]);
        setLoading(false);
        return;
      }

      setResults(data);

    } catch (err) {
      console.error("Fetch error:", err);
      setResults([]);
    }

    setLoading(false);
  };

  // 🔥 Recommendation label logic
  const getRecommendationLabel = (score: number) => {
    if (score >= 0.75)
      return { text: "🔥 Highly Recommended", color: "text-green-400" };
    if (score >= 0.6)
      return { text: "👍 Good Match", color: "text-blue-400" };
    if (score >= 0.45)
      return { text: "😐 Average", color: "text-yellow-400" };
    return { text: "⚠️ Low Match", color: "text-red-400" };
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0f172a] via-[#1e293b] to-[#020617] text-white px-10 py-12">
      
      {/* HEADER */}
      <h1 className="text-5xl font-bold text-center mb-12 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
        🎬 AI Movie Recommender
      </h1>

      {/* SEARCH */}
      <div className="flex justify-center items-center gap-4 mb-14">
        <input
          type="text"
          placeholder="Search for a movie..."
          value={movie}
          onChange={(e) => setMovie(e.target.value)}
          className="w-[420px] px-5 py-3 rounded-xl bg-white/10 backdrop-blur-lg border border-white/20 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
        />

        <button
          onClick={getRecommendations}
          className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 transition shadow-lg hover:shadow-blue-500/30"
        >
          Recommend
        </button>
      </div>

      {/* LOADING */}
      {loading && (
        <p className="text-center text-lg animate-pulse mb-6">
          Fetching recommendations...
        </p>
      )}

      {/* EMPTY STATE */}
      {!loading && results.length === 0 && (
        <p className="text-center text-gray-400 mb-6">
          No recommendations yet. Try searching a movie like "Titanic" or "Toy Story".
        </p>
      )}

      {/* SECTION TITLE */}
      {results.length > 0 && (
        <h2 className="text-2xl font-semibold mb-6 text-blue-300">
          Recommended Movies
        </h2>
      )}

      {/* GRID */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-8">

        {results.map((movie, index) => {
          const label = getRecommendationLabel(movie.ranking_score);

          return (
            <div
              key={index}
              className="group relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5 transition duration-300 shadow-lg hover:scale-105 hover:shadow-blue-500/20"
            >
              {/* glow line */}
              <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-blue-500 to-purple-500 opacity-60 rounded-t-2xl"></div>

              <div className="relative z-10">
                {/* TITLE */}
                <h3 className="text-lg font-semibold mb-3">
                  {movie.title}
                </h3>

                {/* SCORES */}
                <div className="flex flex-col gap-3">

                  {/* ⭐ Rating */}
                  <span className="px-3 py-1 text-sm rounded-full bg-yellow-400/20 text-yellow-300 w-fit">
                    ⭐ {(movie.ranking_score * 5).toFixed(1)} / 5
                  </span>

                  {/* 🔥 Label */}
                  <span className={`text-sm font-medium ${label.color}`}>
                    {label.text}
                  </span>

                  {/* 💬 Sentiment */}
                  <span className="px-3 py-1 text-sm rounded-full bg-purple-400/20 text-purple-300 w-fit">
                    💬 Sentiment: {movie.sentiment_score.toFixed(2)}
                  </span>

                </div>
              </div>
            </div>
          );
        })}

      </div>
    </main>
  );
}