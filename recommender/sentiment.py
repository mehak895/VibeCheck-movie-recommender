import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def _ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


class SentimentAnalyzer:
    def __init__(self, reviews_path):
        _ensure_vader()

        self.reviews = pd.read_csv(reviews_path)
        self.analyzer = SentimentIntensityAnalyzer()

        # Global sentiment prior (used when no movie reviews exist)
        self.global_mean_sentiment = self._compute_global_mean_sentiment()

    def _compute_global_mean_sentiment(self):
        """
        Computes average sentiment across all reviews.
        Acts as a weak prior when movie-level data is missing.
        """
        if self.reviews.empty:
            return 0.1  # weak positive prior

        scores = [
            self.analyzer.polarity_scores(text)["compound"]
            for text in self.reviews["review_text"]
        ]

        return sum(scores) / len(scores)

    def get_sentiment_score(self, movie_id):
        movie_reviews = self.reviews[
            self.reviews["movieId"] == movie_id
        ]

        # Use global prior if no reviews exist
        if movie_reviews.empty:
            return self.global_mean_sentiment

        scores = [
            self.analyzer.polarity_scores(text)["compound"]
            for text in movie_reviews["review_text"]
        ]

        return sum(scores) / len(scores)
