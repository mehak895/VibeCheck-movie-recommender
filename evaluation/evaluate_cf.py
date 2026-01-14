import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from recommender.collaborative import CollaborativeRecommender

# Load ratings
ratings = pd.read_csv("data/ratings.csv")

# Train-test split
train, test = train_test_split(
    ratings, test_size=0.2, random_state=42
)

# Save training data temporarily
train.to_csv("data/train_ratings.csv", index=False)

# Train model on training set
model = CollaborativeRecommender("data/train_ratings.csv")

# Evaluate on test set
squared_errors = []

for _, row in test.iterrows():
    pred = model.predict_rating(row["userId"], row["movieId"])
    squared_errors.append((row["rating"] - pred) ** 2)

rmse = np.sqrt(np.mean(squared_errors))

print("Phase 6 Evaluation Result")
print("--------------------------")
print("RMSE:", round(rmse, 3))
