import streamlit as st
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.sentiment import SentimentAnalyzer

# ---------------- Page Config ---------------- #

st.set_page_config(
    page_title="CineSense",
    page_icon="🎬",
    layout="wide"
)

# ---------------- Header ---------------- #

st.markdown(
    """
    <h1 style="text-align:center;">🎬 CineSense</h1>
    <p style="text-align:center; color: gray;">
        Hybrid Movie Recommendation System<br>
        Content-Based • Collaborative Filtering • Sentiment Analysis
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- Load Models ---------------- #

@st.cache_resource
def load_content_recommender():
    return ContentBasedRecommender(
        movies_path="data/movies.csv",
        tags_path="data/tags.csv",
    )

@st.cache_resource
def load_collaborative_recommender():
    return CollaborativeRecommender(
        ratings_path="data/ratings.csv"
    )

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentAnalyzer(
        reviews_path="data/reviews.csv"
    )

content_recommender = load_content_recommender()
collab_recommender = load_collaborative_recommender()
sentiment_analyzer = load_sentiment_analyzer()

# ---------------- Sidebar (Inputs) ---------------- #

movie_list = sorted(content_recommender.movies["title"].unique())

with st.sidebar:
    st.header("🎯 Recommendation Settings")

    selected_movie = st.selectbox(
        "Choose a movie you like",
        movie_list
    )

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        step=1,
        help="Use a large number (e.g. 999) to simulate a new user"
    )

    recommend_btn = st.button("🚀 Recommend Movies")

# ---------------- Recommendation Logic ---------------- #

if recommend_btn:
    # Step 1: Content-based candidates
    similar_movies = content_recommender.recommend(
        selected_movie, top_n=20
    )

    # Force include selected movie (demo clarity)
    if selected_movie not in similar_movies:
        similar_movies = [selected_movie] + similar_movies

    candidate_movies = content_recommender.movies[
        content_recommender.movies["title"].isin(similar_movies)
    ][["movieId", "title"]]

    predictions = []

    # Step 2: Cold-start user detection
    is_cold_user = not collab_recommender.has_user_history(user_id)

    for _, row in candidate_movies.iterrows():
        sentiment_score = sentiment_analyzer.get_sentiment_score(
            row["movieId"]
        )

        if is_cold_user:
            final_score = sentiment_score
        else:
            cf_score = collab_recommender.predict_rating(
                user_id, row["movieId"]
            )
            final_score = 0.7 * cf_score + 0.3 * sentiment_score

        predictions.append(
            (row["title"], final_score, sentiment_score)
        )

    predictions.sort(key=lambda x: x[1], reverse=True)

    # ---------------- Output ---------------- #

    st.markdown("## 🍿 Personalized Recommendations")

    if is_cold_user:
        st.info(
            "👤 New user detected. Recommendations are based on movie content and sentiment."
        )

    for i, (title, final_score, sentiment_score) in enumerate(
        predictions[:10], start=1
    ):
        st.markdown(
            f"""
            **{i}. {title}**  
            ⭐ Score: `{round(final_score, 2)}`  
            💬 Sentiment: `{round(sentiment_score, 2)}`
            """
        )
