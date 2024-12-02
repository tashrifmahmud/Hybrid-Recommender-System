# Recommendation functions
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
            idx = anime_df.index[anime_df['anime_id'] == anime_id][0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in sim_scores[1:top_n+1]]
            similar_anime = anime_df.iloc[top_indices][['anime_id', 'name', 'genres', 'year', 'studios']]
            similar_anime['similarity_score'] = [sim_scores[i][1] for i in range(1, top_n+1)]
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


def diversify_recommendations_by_keyword(df, column='name', max_per_keyword=2):
    keyword_counts = {}
    diversified = []
    for _, row in df.iterrows():
        name = row[column]
        keywords = [word.lower() for word in name.split() if len(word) > 3]
        if all(keyword_counts.get(keyword, 0) < max_per_keyword for keyword in keywords):
            diversified.append(row)
            for keyword in keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    return pd.DataFrame(diversified)

def fetch_anime_details_v4(anime_name):
    global last_request_time
    base_url = "https://api.jikan.moe/v4"
    search_url = f"{base_url}/anime"
    
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < 1: 
        sleep(1 - time_since_last_request)
    
    try:
        response = requests.get(search_url, params={"q": anime_name, "limit": 1})
        last_request_time = time.time()  
        response.raise_for_status() 
        data = response.json()
        if data and "data" in data and len(data["data"]) > 0:
            anime = data["data"][0]  
            return {
                "title": anime["title"],
                "image_url": anime["images"]["jpg"]["image_url"],
                "synopsis": anime.get("synopsis", "No synopsis available."),
                "score": anime.get("score", "N/A"),
                "url": anime.get("url", "#")
            }
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching details for {anime_name}: {e}")
        return None