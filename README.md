# 🎬 VibeCheck – AI Movie Recommendation System

VibeCheck is a **full-stack hybrid movie recommendation system** that combines
**content similarity**, **collaborative filtering**, and **sentiment analysis** to deliver more accurate and personalized movie suggestions.

Unlike traditional recommenders, VibeCheck integrates **review sentiment signals**, improving ranking quality and aligning recommendations with user perception—not just ratings.

---

## 🚀 Live Demo

👉 **Frontend (Vercel):**
https://vibe-check-movie-recommender-ij03rzw0r-mehak895s-projects.vercel.app

👉 **Backend API (Render):**
https://vibecheck-movie-recommender.onrender.com

---

## ✨ Features

* 🎯 **Content-Based Filtering**
  Uses genres and user-generated tags to compute similarity between movies

* 👥 **Collaborative Filtering (SVD)**
  Learns user behavior patterns from rating data

* 💬 **Sentiment-Aware Ranking**
  Enhances recommendations using sentiment scores from reviews

* ⚖️ **Hybrid Recommendation Engine**
  Combines:

  * Content similarity
  * Collaborative filtering score
  * Sentiment score
  * Popularity

* 🧊 **Cold Start Handling**
  Works even for users with no prior history

* 🌐 **Full-Stack Deployment**

  * Backend: FastAPI (Render)
  * Frontend: Next.js + Tailwind (Vercel)

---

## 🛠 Tech Stack

### Backend

* Python
* FastAPI
* Pandas, NumPy
* Scikit-learn (CountVectorizer, TruncatedSVD)
* NLTK (VADER Sentiment Analyzer)
* Custom trained sentiment model (IMDB dataset)

### Frontend

* Next.js (React)
* Tailwind CSS

### Deployment

* Render (Backend API)
* Vercel (Frontend UI)

---

## 📊 Dataset

* MovieLens Dataset:

  * movies
  * ratings
  * tags

* IMDB Review Dataset:

  * Used to train sentiment classification model

---

## 🧠 How It Works

1. **Content-Based Module**
   Builds similarity using genres + tags

2. **Collaborative Filtering**
   Applies matrix factorization (SVD) on user-item interactions

3. **Sentiment Analysis**
   Uses a trained model to compute sentiment scores from reviews

4. **Final Ranking Engine**

   Final score combines multiple signals:

   ```
   Ranking Score = CF Score + Similarity Score + Sentiment Score + Popularity Score
   ```

---

## 🔗 API Usage

Example:

```bash
GET /recommend?movie=Titanic&user_id=3
```

Returns:

```json
[
  {
    "title": "Some Movie",
    "ranking_score": 0.67,
    "sentiment_score": 0.82
  }
]
```

---

## ▶️ Run Locally

### Backend

```bash
pip install -r requirements.txt
python backend/main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## ⚠️ Notes

* Backend may take **20–40 seconds on first request** due to cold start (Render free tier)
* Sentiment model is loaded from serialized `.pkl` files

---

## 📌 Future Improvements

* 🔍 Smart search with autocomplete (Netflix-style)
* 🎞 Movie posters using TMDB API
* 👤 User authentication & personalization
* 📊 Explainable recommendations (“Why this movie?”)
* ⚡ Performance optimization & caching

---

## 💡 Key Highlight

> This project demonstrates how **sentiment analysis can enhance traditional recommender systems**, making recommendations more aligned with real user opinions rather than just numerical ratings.

---

## 👩‍💻 Author

**Mehak Mittal**
