from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.sentiment import SentimentAnalyzer
from recommender.engine import RecommendationEngine

import os

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for hackathon/demo (later restrict to Vercel URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 Lazy load
engine = None


def get_engine():
    global engine

    if engine is None:
        try:
            print("🔥 Loading models...")

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            content = ContentBasedRecommender(
                os.path.join(BASE_DIR, "../data/movies_small.csv"),
                os.path.join(BASE_DIR, "../data/tags.csv")
            )

            cf = CollaborativeRecommender(
                os.path.join(BASE_DIR, "../data/ratings_small.csv")
            )

            sentiment = SentimentAnalyzer()

            engine = RecommendationEngine(content, cf, sentiment)

            print("✅ Models loaded successfully")

        except Exception as e:
            print("❌ ERROR while loading engine:", str(e))
            raise e

    return engine


@app.get("/")
def home():
    return {"message": "Movie Recommender API is running"}


@app.get("/recommend")
def recommend(movie: str, user_id: int = 3):
    try:
        eng = get_engine()

        results = eng.recommend(user_id, movie)

        if not results:
            return {"message": "No recommendations found"}

        return results

    except Exception as e:
        print("❌ ERROR in /recommend:", str(e))
        raise HTTPException(status_code=500, detail=str(e))