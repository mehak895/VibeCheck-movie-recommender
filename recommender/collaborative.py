import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    def __init__(self, ratings_path):
        self.ratings = pd.read_csv(ratings_path)
        self._prepare_matrix()
        self._train_model()

    def _prepare_matrix(self):
        # Create user-movie matrix
        self.user_movie_matrix = self.ratings.pivot(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

    def _train_model(self):
        # Matrix factorization using SVD
        self.svd = TruncatedSVD(n_components=20, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_movie_matrix)
        self.movie_factors = self.svd.components_

    def has_user_history(self, user_id):
        # Phase 5: cold-start detection
        return user_id in self.user_movie_matrix.index

    def predict_rating(self, user_id, movie_id):
        if user_id not in self.user_movie_matrix.index:
            return 0.0
        if movie_id not in self.user_movie_matrix.columns:
            return 0.0

        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)

        # Dot product gives predicted rating
        return float(
            np.dot(
                self.user_factors[user_idx],
                self.movie_factors[:, movie_idx]
            )
        )
