from fastapi import FastAPI
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.sentiment import SentimentAnalyzer
from recommender.engine import RecommendationEngine

app = FastAPI()

# 🚀 Lazy load (initially None)
engine = None


def get_engine():
    global engine

    if engine is None:
        print("🔥 Loading models...")

        content = ContentBasedRecommender(
            "data/movies_small.csv",
            "data/tags.csv"
        )

        cf = CollaborativeRecommender(
            "data/ratings_small.csv"
        )

        sentiment = SentimentAnalyzer()

        engine = RecommendationEngine(content, cf, sentiment)

    return engine


@app.get("/")
def home():
    return {"message": "Movie Recommender API is running"}


@app.get("/recommend")
def recommend(movie: str, user_id: int = 3):
    eng = get_engine()  # 🔥 load here

    results = eng.recommend(user_id, movie)

    return results