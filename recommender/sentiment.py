import pickle


class SentimentAnalyzer:
    def __init__(self):
        # Load trained model + vectorizer
        with open("models/sentiment_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open("models/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def get_sentiment_score(self, text):
        """
        Predict sentiment from text (movie content)
        """

        if not text:
            return 0.0

        X = self.vectorizer.transform([text])
        prob = self.model.predict_proba(X)[0][1]  # positive prob

        # Convert [0,1] → [-1,1]
        score = (prob * 2) - 1
        score = score ** 3

        return float(score)