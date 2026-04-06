import pickle
import os


class SentimentAnalyzer:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(BASE_DIR, "../models/sentiment_model.pkl")
        vectorizer_path = os.path.join(BASE_DIR, "../models/vectorizer.pkl")

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)

            print("✅ Sentiment model loaded")

        except Exception as e:
            print("❌ Error loading sentiment model:", str(e))
            self.model = None
            self.vectorizer = None

    def predict(self, text):
        """
        Returns sentiment score between 0 and 1
        """
        if self.model is None or self.vectorizer is None:
            return 0.5  # fallback neutral

        try:
            vec = self.vectorizer.transform([text])
            prob = self.model.predict_proba(vec)[0][1]
            return float(prob)

        except Exception as e:
            print("❌ Sentiment prediction error:", str(e))
            return 0.5