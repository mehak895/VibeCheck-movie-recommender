# CineSense – Movie Recommendation System

CineSense is a content-based movie recommendation system that suggests movies similar to a selected title based on metadata such as genre, keywords, and overview.  
The project is structured using a **3-layer design** to keep logic clean, maintainable, and easy to extend.

---

## How the Recommendation Works

1. Movie metadata (genres, keywords, overview, cast) is combined into a single feature set  
2. Text data is vectorized using TF-IDF  
3. Cosine similarity is used to find movies most similar to the selected one  
4. Top matching movies are returned as recommendations  

The approach is **explainable** and easy to debug compared to black-box models.

---

## 3-Layer Project Structure (Important)

The project is intentionally divided into three layers:

### 1️⃣ Data Layer
- Loads and preprocesses the movie dataset  
- Cleans missing values and prepares features  
- Responsible only for data handling  

### 2️⃣ Recommendation / Logic Layer
- Builds the similarity model  
- Computes similarity scores  
- Contains all recommendation logic  
- Independent of UI or presentation  

### 3️⃣ Presentation Layer
- Streamlit-based UI  
- Takes user input (movie selection)  
- Displays recommended movies  
- Does not contain business logic  

This separation ensures:
- Logic can be reused without the UI  
- UI changes do not affect recommendation logic  
- Easier debugging and future extensions  

---

## Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn  
- **Recommendation:** TF-IDF, Cosine Similarity  
- **UI:** Streamlit  

---

## How to Run the Project

```bash
git clone https://github.com/mehak895/cinesense-movie-recommender.git
cd cinesense-movie-recommender
pip install -r requirements.txt
streamlit run app.py

