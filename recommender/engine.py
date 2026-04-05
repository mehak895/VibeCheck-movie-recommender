class RecommendationEngine:
    """
    Backend ranking engine that fuses multiple recommendation signals.
    """

    def __init__(
        self,
        content_recommender,
        collaborative_recommender,
        sentiment_analyzer,
    ):
        self.content = content_recommender
        self.cf = collaborative_recommender
        self.sentiment = sentiment_analyzer

    def _normalize(self, scores):
        """
        Min-max normalization
        """
        if not scores:
            return scores

        min_s = min(scores)
        max_s = max(scores)

        if max_s - min_s == 0:
            return [0.5] * len(scores)

        return [(s - min_s) / (max_s - min_s) for s in scores]

    def recommend(
        self,
        user_id,
        movie_title,
        top_k=10,
        candidate_pool=30,
        use_sentiment=True,
    ):
        """
        Generates ranked movie recommendations for a user.
        """

        # Step 1: Candidate generation
        candidates = self.content.generate_candidates(
            movie_title, top_n=candidate_pool
        )

        if not candidates:
            return []

        is_cold_user = not self.cf.has_user_history(user_id)

        cf_scores = []
        sim_scores = []
        sent_scores = []
        pop_scores = []

        movie_data = []

        # Step 2: Collect all raw scores
        for candidate in candidates:
            movie_id = candidate["movieId"]
            title = candidate["title"]
            similarity_score = candidate["similarity_score"]

            # 🔥 Get movie content
            movie_row = self.content.movies[
                self.content.movies["movieId"] == movie_id
            ]

            if not movie_row.empty:
                genres_tags = movie_row["content"].values[0].lower()
            else:
                genres_tags = ""

            title_lower = title.lower()

            # 🔥 IMPORTANT FIX: create rich sentence
            content_text = f"This is a {genres_tags} movie called {title_lower}. It is a {genres_tags} film."

            # Sentiment from ML model
            sentiment_score = self.sentiment.get_sentiment_score(content_text)

            # Popularity
            popularity_score = self.cf.get_popularity_score(movie_id)

            # Collaborative score
            if is_cold_user:
                cf_score = 0.0
            else:
                cf_score = self.cf.get_affinity_score(user_id, movie_id)

            cf_scores.append(cf_score)
            sim_scores.append(similarity_score)
            sent_scores.append(sentiment_score)
            pop_scores.append(popularity_score)

            movie_data.append(
                {
                    "movieId": movie_id,
                    "title": title,
                }
            )

        # Step 3: Normalize all scores
        cf_scores = self._normalize(cf_scores)
        sim_scores = self._normalize(sim_scores)
        sent_scores = self._normalize(sent_scores)
        pop_scores = self._normalize(pop_scores)

        # Step 4: Final ranking
        ranked_results = []

        for i in range(len(movie_data)):
            score = (
                0.4 * cf_scores[i] +
                0.3 * sim_scores[i] +
                0.2 * sent_scores[i] +
                0.1 * pop_scores[i]
            )

            ranked_results.append(
                {
                    "movieId": movie_data[i]["movieId"],
                    "title": movie_data[i]["title"],
                    "ranking_score": score,
                    "cf_score": cf_scores[i],
                    "similarity_score": sim_scores[i],
                    "sentiment_score": sent_scores[i],
                    "popularity_score": pop_scores[i],
                }
            )

        # Step 5: Sort
        ranked_results.sort(
            key=lambda x: x["ranking_score"], reverse=True
        )

        return ranked_results[:top_k]