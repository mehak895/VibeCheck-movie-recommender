import pickle


class SentimentAnalyzer:
    def __init__(self):
        # Load trained model
        with open("models/sentiment_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open("models/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def _predict_sentiment(self, texts):
        if not texts:
            return [0.0]

        X = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(X)[:, 1]

        # Convert [0,1] → [-1,1]
        scores = (probs * 2) - 1
        return scores

    def get_sentiment_score(self, text):
        scores = self._predict_sentiment([text])
        return float(scores[0])