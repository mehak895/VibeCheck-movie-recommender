# VibeCheck – Movie Recommendation System

VibeCheck is a backend-focused hybrid movie recommendation system that combines  
**content similarity**, **user behavior**, and **review sentiment** to rank movies more effectively.

The system is designed to demonstrate how sentiment analysis can improve the quality
of traditional recommendation pipelines.

---

## Features

- Content-based recommendations using movie genres and user-generated tags  
- Collaborative filtering using matrix factorization (SVD) on user ratings  
- Sentiment-aware ranking using review text (VADER sentiment analysis)  
- Cold-start handling for new users with no rating history  
- Offline evaluation to compare recommendations **with and without sentiment**

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn (CountVectorizer, TruncatedSVD)  
- NLTK (VADER Sentiment Analyzer)

---

## Dataset

- MovieLens (movies, ratings, tags)  
- Custom review dataset for sentiment analysis

---

## How to Run (Backend)

```bash
pip install -r requirements.txt
python run_backend.py
