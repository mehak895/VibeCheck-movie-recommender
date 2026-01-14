import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self, reviews_path):
        self.reviews = pd.read_csv(reviews_path)
        self.analyzer = SentimentIntensityAnalyzer()
        self.movie_sentiment = self._compute_movie_sentiment()

    def _compute_movie_sentiment(self):
        # Compute sentiment score per review
        self.reviews["sentiment"] = self.reviews["review_text"].apply(
            lambda x: self.analyzer.polarity_scores(str(x))["compound"]
        )

        # Aggregate sentiment per movie
        movie_sentiment = (
            self.reviews.groupby("movieId")["sentiment"]
            .mean()
            .to_dict()
        )
        return movie_sentiment

    def get_sentiment_score(self, movie_id):
        # Neutral fallback for missing reviews
        return self.movie_sentiment.get(movie_id, 0.0)
