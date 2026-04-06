import pickle

class SentimentAnalyzer:
    def __init__(self):
        try:
            with open("models/vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)

            with open("models/sentiment_model.pkl", "rb") as f:
                self.model = pickle.load(f)

        except:
            print("⚠️ Sentiment model not found, using default")
            self.vectorizer = None
            self.model = None

    def get_sentiment_score(self, text):
        """
        Returns sentiment score between 0 and 1
        """

        if self.model is None or self.vectorizer is None:
            return 0.5  # fallback neutral

        try:
            vec = self.vectorizer.transform([text])
            score = self.model.predict_proba(vec)[0][1]
            return float(score)
        except:
            return 0.5