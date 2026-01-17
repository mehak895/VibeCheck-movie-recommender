import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.sentiment import SentimentAnalyzer
from recommender.engine import RecommendationEngine


# ---------------- Load Data ---------------- #

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

RELEVANT_RATING = 4.0
TOP_K = 10

# Sample users (limit for faster evaluation)
users = ratings["userId"].unique()[:50]

# ---------------- Initialize Models ---------------- #

content = ContentBasedRecommender(
    "data/movies.csv", "data/tags.csv"
)
collab = CollaborativeRecommender("data/ratings.csv")
sentiment = SentimentAnalyzer("data/reviews.csv")

engine = RecommendationEngine(content, collab, sentiment)


# ---------------- Metrics ---------------- #

def precision_recall_at_k(recommended, relevant):
    recommended = set(recommended)
    relevant = set(relevant)

    tp = len(recommended & relevant)

    precision = tp / len(recommended) if recommended else 0
    recall = tp / len(relevant) if relevant else 0

    return precision, recall


def average_sentiment_at_k(recommendations):
    if not recommendations:
        return 0.0
    sentiments = [r["sentiment_score"] for r in recommendations]
    return np.mean(sentiments)


# ---------------- Evaluation ---------------- #

prec_no_sent, rec_no_sent = [], []
prec_sent, rec_sent = [], []

sent_no_sent, sent_with_sent = [], []

for user_id in users:
    user_ratings = ratings[ratings["userId"] == user_id]
    high_rated = user_ratings[
        user_ratings["rating"] >= RELEVANT_RATING
    ]["movieId"].tolist()

    if len(high_rated) < 2:
        continue

    # Use a known liked movie as seed
    seed_movie_id = high_rated[0]
    seed_title = movies[
        movies["movieId"] == seed_movie_id
    ]["title"].values[0]

    # -------- WITHOUT Sentiment -------- #

    recs_no_sent = engine.recommend(
        user_id, seed_title, use_sentiment=False
    )

    rec_ids_no_sent = [r["movieId"] for r in recs_no_sent]

    p, r = precision_recall_at_k(rec_ids_no_sent, high_rated)
    prec_no_sent.append(p)
    rec_no_sent.append(r)

    sent_no_sent.append(
        average_sentiment_at_k(recs_no_sent)
    )

    # -------- WITH Sentiment -------- #

    recs_sent = engine.recommend(
        user_id, seed_title, use_sentiment=True
    )

    rec_ids_sent = [r["movieId"] for r in recs_sent]

    p, r = precision_recall_at_k(rec_ids_sent, high_rated)
    prec_sent.append(p)
    rec_sent.append(r)

    sent_with_sent.append(
        average_sentiment_at_k(recs_sent)
    )


# ---------------- Results ---------------- #

print("\nHybrid Recommendation Evaluation (Top-10)")
print("========================================")

print("\nWITHOUT Sentiment:")
print("  Precision@10       =", round(np.mean(prec_no_sent), 3))
print("  Recall@10          =", round(np.mean(rec_no_sent), 3))
print("  Avg Sentiment@10   =", round(np.mean(sent_no_sent), 3))

print("\nWITH Sentiment:")
print("  Precision@10       =", round(np.mean(prec_sent), 3))
print("  Recall@10          =", round(np.mean(rec_sent), 3))
print("  Avg Sentiment@10   =", round(np.mean(sent_with_sent), 3))

print("\nΔ Improvement:")
print(
    "  Precision          =",
    round(np.mean(prec_sent) - np.mean(prec_no_sent), 3)
)
print(
    "  Recall             =",
    round(np.mean(rec_sent) - np.mean(rec_no_sent), 3)
)
print(
    "  Sentiment Gain     =",
    round(np.mean(sent_with_sent) - np.mean(sent_no_sent), 3)
)
