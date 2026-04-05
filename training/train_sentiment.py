import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def main():
    print("Loading dataset...")

    df = pd.read_csv("data/IMDB Dataset.csv")

    # Convert labels to 0/1
    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    X = df["review"]
    y = df["sentiment"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    accuracy = model.score(X_test_vec, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")

    print("Saving model...")

    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()