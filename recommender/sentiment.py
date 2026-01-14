import pandas as pd
import nltk
import os
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

    def get_sentiment_score(self, movie_id):
        movie_reviews = self.reviews[
            self.reviews["movieId"] == movie_id
        ]

        if movie_reviews.empty:
            return 0.0

        scores = [
            self.analyzer.polarity_scores(text)["compound"]
            for text in movie_reviews["review_text"]
        ]

        return sum(scores) / len(scores)
