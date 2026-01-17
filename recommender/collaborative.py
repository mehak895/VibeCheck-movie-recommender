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

    def _prepare_interaction_matrix(self):
        """
        Builds the user–movie interaction matrix.
        Rows: users
        Columns: movies
        """
        self.user_movie_matrix = self.ratings.pivot(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

    def _train_factor_model(self):
        """
        Learns latent user and movie factors using SVD.
        """
        self.svd = TruncatedSVD(n_components=20, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_movie_matrix)
        self.movie_factors = self.svd.components_

    def has_user_history(self, user_id):
        """
        Checks whether the user exists in historical data.
        Used for cold-start detection.
        """
        return user_id in self.user_movie_matrix.index

    def get_affinity_score(self, user_id, movie_id):
        """
        Returns a predicted affinity score between a user and a movie.
        This is NOT a final recommendation score.
        """
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
