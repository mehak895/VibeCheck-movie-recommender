class RecommendationEngine:
    """
    Backend ranking engine that fuses multiple recommendation signals.
    """

    def __init__(
        self,
        content_recommender,
        collaborative_recommender,
        sentiment_analyzer,
        cf_weight=0.7,
        sentiment_weight=0.3,
    ):
        self.content = content_recommender
        self.cf = collaborative_recommender
        self.sentiment = sentiment_analyzer

        self.cf_weight = cf_weight
        self.sentiment_weight = sentiment_weight

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

        # Step 1: Content-based candidate generation
        candidates = self.content.generate_candidates(
            movie_title, top_n=candidate_pool
        )

        if not candidates:
            return []

        # Step 2: Cold-start detection
        is_cold_user = not self.cf.has_user_history(user_id)

        ranked_results = []

        for candidate in candidates:
            movie_id = candidate["movieId"]
            title = candidate["title"]
            similarity_score = candidate["similarity_score"]

            # Always compute sentiment (for transparency)
            sentiment_score = self.sentiment.get_sentiment_score(movie_id)

            if is_cold_user:
                ranking_score = similarity_score
                if use_sentiment:
                    ranking_score += sentiment_score
            else:
                cf_score = self.cf.get_affinity_score(user_id, movie_id)
                ranking_score = self.cf_weight * cf_score
                if use_sentiment:
                    ranking_score += self.sentiment_weight * sentiment_score

            ranked_results.append(
                {
                    "movieId": movie_id,
                    "title": title,
                    "ranking_score": ranking_score,
                    "sentiment_score": sentiment_score,
                }
            )

        # Step 3: Sort by ranking score
        ranked_results.sort(
            key=lambda x: x["ranking_score"], reverse=True
        )

        return ranked_results[:top_k]
