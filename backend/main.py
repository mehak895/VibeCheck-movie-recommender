from fastapi import FastAPI
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.sentiment import SentimentAnalyzer
from recommender.engine import RecommendationEngine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once (important)
content = ContentBasedRecommender("data/movies.csv", "data/tags.csv")
cf = CollaborativeRecommender("data/ratings.csv")
sentiment = SentimentAnalyzer()

engine = RecommendationEngine(content, cf, sentiment)


@app.get("/")
def home():
    return {"message": "Movie Recommender API is running"}


@app.get("/recommend")
def recommend(movie: str, user_id: int = 1):
    results = engine.recommend(
        user_id=user_id,
        movie_title=movie,
        top_k=10
    )
    return results