import pandas as pd

df = pd.read_csv("data/movies.csv")
df.head(3000).to_csv("data/movies_small.csv", index=False)

df = pd.read_csv("data/ratings.csv")
df.head(20000).to_csv("data/ratings_small.csv", index=False)

print("✅ Done")