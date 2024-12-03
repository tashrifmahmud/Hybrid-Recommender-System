import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
from jikanpy import Jikan

# Tab name and icon
st.set_page_config(
    page_title="Anime Recommender", 
    page_icon="https://i.imgur.com/eDEuX3a.jpeg",  
    layout="wide",  
    initial_sidebar_state="expanded"  
)

# Title
st.title(":tv: Hybrid Anime Recommendation System")
st.markdown(":cat: [GitHub Repository](https://github.com/tashrifmahmud/Hybrid-Recommender-System)  | :sparkles: [Jikan API](https://jikan.moe/)")

# Banner
st.image("https://i.imgur.com/IhTFtPw.jpeg", use_container_width=True)

# Sidebar
with st.sidebar:
    st.header("More about this Project:")
    st.markdown("### :space_invader: Created by: Tashrif Mahmud\n- This Anime Recommendation System combines collaborative and content-based filtering to provide personalized anime suggestions. Explore detailed recommendations with images, scores, and synopses powered by Jikan API.")
    st.markdown("### :link: Links:\n- :cat: [GitHub](https://github.com/tashrifmahmud/Hybrid-Recommender-System)\n- :e-mail: [LinkedIn](https://www.linkedin.com/in/tashrifmahmud/)") 

# Github release file links
file_urls = {
    "anime_filtered_processed_st.csv": "https://github.com/tashrifmahmud/Hybrid-Recommender-System/releases/download/v1.0.0/anime_filtered_processed_st.csv",
    "cosine_sim_reduced.npy": "https://github.com/tashrifmahmud/Hybrid-Recommender-System/releases/download/v1.0.0/cosine_sim_reduced.npy",
    "svd_model_3.pkl": "https://github.com/tashrifmahmud/Hybrid-Recommender-System/releases/download/v1.0.0/svd_model_3.pkl",
    "user_clean_processed_2.csv": "https://github.com/tashrifmahmud/Hybrid-Recommender-System/releases/download/v1.0.0/user_clean_processed_2.csv"
}

# Function to download files with error handling
def download_file(url, output_path):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"File download failed for {url}: {e}")
        return False
    return True

st.info("Initial loading can take a few minutes, thank you for your patience.", icon="ℹ️")

# Load all saved data with caching
@st.cache_data
def load_data():
    download_file(file_urls["anime_filtered_processed_st.csv"], "anime_filtered_processed_st.csv")
    anime_filtered_df = pd.read_csv("anime_filtered_processed_st.csv")

    download_file(file_urls["cosine_sim_reduced.npy"], "cosine_sim_reduced.npy")
    cosine_sim = np.load("cosine_sim_reduced.npy")

    download_file(file_urls["svd_model_3.pkl"], "svd_model_3.pkl")
    svd = joblib.load("svd_model_3.pkl")

    download_file(file_urls["user_clean_processed_2.csv"], "user_clean_processed_2.csv")
    user_clean = pd.read_csv("user_clean_processed_2.csv")

    return anime_filtered_df, cosine_sim, svd, user_clean

# Load data
anime_filtered_df, cosine_sim, svd, user_clean = load_data()

st.success("Data loaded successfully!", icon="✅")

# Recommendation Functions
def hybrid_recommendations_for_new_user(new_user_ratings, svd_model, cosine_sim, anime_df, top_n=40, cf_weight=0.4, content_weight=0.6):
    collaborative_scores = []
    all_anime_ids = anime_df['anime_id'].unique()

    # Collaborative filtering predictions
    for anime_id in all_anime_ids:
        try:
            pred = svd_model.predict(uid=0, iid=anime_id)
            collaborative_scores.append((anime_id, pred.est))
        except Exception:
            collaborative_scores.append((anime_id, 0))
    collaborative_df = pd.DataFrame(collaborative_scores, columns=['anime_id', 'cf_score'])

    # Content-based scores
    watched_anime_ids = [rating['anime_id'] for rating in new_user_ratings]
    content_scores = []
    for anime_id in watched_anime_ids:
        try:
            idx = anime_df.index[anime_df['anime_id'] == anime_id]
            if len(idx) == 0:
                continue
            idx = idx[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in sim_scores[1:top_n + 1]]
            similar_anime = anime_df.iloc[top_indices][['anime_id', 'name', 'genres', 'year', 'studios']]
            similar_anime['similarity_score'] = [sim_scores[i][1] for i in range(1, top_n + 1)]
            content_scores.append(similar_anime[['anime_id', 'similarity_score']])
        except Exception as e:
            st.write(f"Error processing Anime ID {anime_id}: {e}")
    if content_scores:
        content_scores = pd.concat(content_scores).groupby('anime_id', as_index=False).mean()
    else:
        content_scores = pd.DataFrame(columns=['anime_id', 'similarity_score'])

    # Combine collaborative and content-based scores
    hybrid_df = pd.merge(collaborative_df, content_scores, on='anime_id', how='outer').fillna(0)
    hybrid_df['hybrid_score'] = (cf_weight * hybrid_df['cf_score']) + (content_weight * hybrid_df['similarity_score'])
    hybrid_df = pd.merge(hybrid_df, anime_df[['anime_id', 'name', 'genres', 'year', 'studios']], on='anime_id')
    hybrid_df = hybrid_df.sort_values(by='hybrid_score', ascending=False).head(top_n)
    return hybrid_df

def fetch_anime_details_v4(anime_name, last_request_time):
    base_url = "https://api.jikan.moe/v4"
    search_url = f"{base_url}/anime"

    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < 0.7:
        time.sleep(0.7 - time_since_last_request)

    try:
        response = requests.get(search_url, params={"q": anime_name, "limit": 1}, timeout=10)
        response.raise_for_status()
        last_request_time = time.time()

        data = response.json()
        if data and "data" in data and len(data["data"]) > 0:
            anime = data["data"][0]
            return {
                "title": anime["title"],
                "image_url": anime["images"]["jpg"]["image_url"],
                "synopsis": anime.get("synopsis", "No synopsis available."),
                "score": anime.get("score", "N/A"),
                "url": anime.get("url", "#"),
            }, last_request_time
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching details for {anime_name}: {e}")
        return None, last_request_time

# User input form
last_request_time = 0

with st.form("recommendation_form"):
    new_user_ratings = []
    for i in range(5):
        anime_name = st.selectbox(f"Select Anime {i + 1}", anime_filtered_df['name'].unique(), key=f"anime_{i}")
        rating = st.slider(f"Rate {anime_name} (1 to 10)", 1, 10, 5, key=f"rating_{i}")
        if anime_name:
            anime_id = anime_filtered_df.loc[anime_filtered_df['name'] == anime_name, 'anime_id'].values[0]
            new_user_ratings.append({'user_id': 0, 'anime_id': anime_id, 'rating': rating})
    submit_button = st.form_submit_button(label="Get Recommendations")

if submit_button:
    if len(new_user_ratings) == 0:
        st.write("Please select at least one anime and provide ratings.")
    else:
        hybrid_recs = hybrid_recommendations_for_new_user(
            new_user_ratings=new_user_ratings,
            svd_model=svd,
            cosine_sim=cosine_sim,
            anime_df=anime_filtered_df,
            top_n=40,
            cf_weight=0.4,
            content_weight=0.6
        )
        st.write("Hybrid Recommendations:")
        for _, row in hybrid_recs.iterrows():
            anime_details, last_request_time = fetch_anime_details_v4(row['name'], last_request_time)
            if anime_details:
                st.image(anime_details['image_url'], use_container_width=True)
                st.markdown(f"### [{anime_details['title']}]({anime_details['url']})")
                st.markdown(f"**Score**: {anime_details['score']}")
                st.markdown(f"**Genres**: {row['genres']}")
                st.markdown(f"**Synopsis**: {anime_details['synopsis']}")
                st.markdown("---")
