import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Backend module responsible for content-based candidate generation.
    """

    def __init__(self, movies_path, tags_path):
        self.movies = pd.read_csv(movies_path)
        self.tags = pd.read_csv(tags_path)

        self._prepare_features()
        self._build_similarity_matrix()

    def _prepare_features(self):
        """
        Combines genres and user-generated tags into a single textual feature.
        """
        tags_grouped = (
            self.tags.groupby("movieId")["tag"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )

        self.movies = self.movies.merge(
            tags_grouped, on="movieId", how="left"
        )

        self.movies["tag"] = self.movies["tag"].fillna("")

        self.movies["content"] = (
            self.movies["genres"].str.replace("|", " ", regex=False)
            + " "
            + self.movies["tag"]
        )

    def _build_similarity_matrix(self):
        """
        Builds a cosine similarity matrix over movie content features.
        """
        vectorizer = CountVectorizer(stop_words="english")
        count_matrix = vectorizer.fit_transform(self.movies["content"])
        self.similarity = cosine_similarity(count_matrix)

    def generate_candidates(self, movie_title, top_n=20):
        """
        Generates a candidate set of movies similar to the given movie.

        Returns:
            List of dicts: [
                {
                    "movieId": int,
                    "title": str,
                    "similarity_score": float
                }
            ]
        """
        if movie_title not in self.movies["title"].values:
            return []

        idx = self.movies[self.movies["title"] == movie_title].index[0]

        similarity_scores = list(enumerate(self.similarity[idx]))
        similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True
        )

        candidates = similarity_scores[1 : top_n + 1]

        results = []
        for i, score in candidates:
            row = self.movies.iloc[i]
            results.append(
                {
                    "movieId": int(row["movieId"]),
                    "title": row["title"],
                    "similarity_score": float(score),
                }
            )

        return results
