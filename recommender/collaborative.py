import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    """
    Backend module responsible for computing user–item affinity
    using collaborative filtering (matrix factorization).
    """

    def __init__(self, ratings_path):
        self.ratings = pd.read_csv(ratings_path)
        self._prepare_interaction_matrix()
        self._train_factor_model()
        self._compute_movie_popularity()  # NEW

    def _prepare_interaction_matrix(self):
        self.user_movie_matrix = self.ratings.pivot(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

    def _train_factor_model(self):
        self.svd = TruncatedSVD(n_components=10, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_movie_matrix)
        self.movie_factors = self.svd.components_

    def _compute_movie_popularity(self):
        """
        Computes popularity score using average rating and number of ratings.
        """
        self.movie_popularity = (
            self.ratings.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )

    def has_user_history(self, user_id):
        return user_id in self.user_movie_matrix.index

    def get_affinity_score(self, user_id, movie_id):
        if user_id not in self.user_movie_matrix.index:
            return 0.0
        if movie_id not in self.user_movie_matrix.columns:
            return 0.0

        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)

        return float(
            np.dot(
                self.user_factors[user_idx],
                self.movie_factors[:, movie_idx]
            )
        )

    def get_popularity_score(self, movie_id):
        """
        Returns popularity score (quality + confidence).
        """
        row = self.movie_popularity[
            self.movie_popularity["movieId"] == movie_id
        ]

        if row.empty:
            return 0.0

        mean_rating = row["mean"].values[0]
        count = row["count"].values[0]

        return float(mean_rating * np.log1p(count))