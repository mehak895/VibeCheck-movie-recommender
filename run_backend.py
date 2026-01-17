from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.sentiment import SentimentAnalyzer
from recommender.engine import RecommendationEngine


def main():
    content = ContentBasedRecommender("data/movies.csv", "data/tags.csv")
    cf = CollaborativeRecommender("data/ratings.csv")
    sentiment = SentimentAnalyzer("data/reviews.csv")

    engine = RecommendationEngine(content, cf, sentiment)

    results = engine.recommend(
        user_id=3,
        movie_title="Titanic (1997)",
        top_k=10,
        use_sentiment=True
    )

    print("\nTop Recommendations:\n")

    for i, rec in enumerate(results, start=1):
        print(
            f"{i}. {rec['title']} | "
            f"Score={rec['ranking_score']:.2f} | "
            f"Sentiment={rec['sentiment_score']:.2f}"
        )


if __name__ == "__main__":
    main()
