# Example: New user ratings
new_user_ratings = [
    {'user_id': 0, 'anime_id': 11111, 'rating': 8},   # Anime ID 11111 with rating 8
    {'user_id': 0, 'anime_id': 5042, 'rating': 9},  # Anime ID 5042 with rating 9
    {'user_id': 0, 'anime_id': 11617, 'rating': 9}, # Anime ID 11617 with rating 9
]

# Convert to DataFrame and append to user_clean
new_user_df = pd.DataFrame(new_user_ratings)
updated_user_clean = pd.concat([user_clean, new_user_df], ignore_index=True)

def get_recommendations_with_score_and_rank(anime_id, cosine_sim, df, top_n=10, popularity_threshold=10000):
    # Find index of the given anime_id
    idx = df.index[df['anime_id'] == anime_id][0]
    
    # Get similarity scores for this anime
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top similar anime indices
    top_anime_indices = [i[0] for i in sim_scores[1:]]
    top_sim_scores = [i[1] for i in sim_scores[1:]]
    
    # Combine results into a DataFrame
    recommendations = df.iloc[top_anime_indices][['anime_id', 'name', 'genres', 'popularity', 'score', 'rank']].copy()
    recommendations['similarity_score'] = top_sim_scores
    
    # Filter based on popularity threshold
    recommendations = recommendations[recommendations['popularity'] <= popularity_threshold]
    
    # Normalize columns for fairness
    recommendations['popularity_norm'] = 1 / (recommendations['popularity'] + 1)
    recommendations['rank_norm'] = 1 / (recommendations['rank'] + 1)
    recommendations['score_norm'] = recommendations['score'] / 10  # Assuming max score is 10
    
    # Weighted score: Adjust weights to balance factors
    recommendations['weighted_score'] = (
        0.9 * recommendations['similarity_score'] +
        0.1 * recommendations['popularity_norm'] +
        0 * recommendations['score_norm'] +
        0 * recommendations['rank_norm']
    )
    
    # Sort by weighted_score
    recommendations = recommendations.sort_values(by='weighted_score', ascending=False)
    
    # Return top_n recommendations
    return recommendations.head(top_n)

def hybrid_recommendations_for_new_user(new_user_ratings, svd_model, cosine_sim, anime_df, top_n=10, cf_weight=0.6, content_weight=0.4):
    collaborative_scores = []
    all_anime_ids = anime_df['anime_id'].unique()

    for anime_id in all_anime_ids:
        try:
            pred = svd_model.predict(uid=0, iid=anime_id)
            collaborative_scores.append((anime_id, pred.est))
        except Exception:
            collaborative_scores.append((anime_id, 0))

    collaborative_df = pd.DataFrame(collaborative_scores, columns=['anime_id', 'cf_score'])

    # Content-Based Recommendations
    watched_anime_ids = [rating['anime_id'] for rating in new_user_ratings]
    content_scores = []

    for anime_id in watched_anime_ids:
        print(f"Fetching recommendations for Anime ID: {anime_id}")
        try:
            similar_anime = get_recommendations_with_score_and_rank(anime_id, cosine_sim, anime_df, top_n=10)
            content_scores.append(similar_anime[['anime_id', 'weighted_score']])
        except Exception as e:
            print(f"Error processing Anime ID {anime_id}: {e}")

    if content_scores:
        content_scores = pd.concat(content_scores).groupby('anime_id', as_index=False).mean()
    else:
        print("No content-based recommendations found. Defaulting to collaborative filtering only.")
        content_scores = pd.DataFrame(columns=['anime_id', 'weighted_score'])

    hybrid_df = pd.merge(collaborative_df, content_scores, on='anime_id', how='outer').fillna(0)
    hybrid_df['hybrid_score'] = (cf_weight * hybrid_df['cf_score']) + (content_weight * hybrid_df['weighted_score'])
    hybrid_df = pd.merge(hybrid_df, anime_df[['anime_id', 'name', 'genres', 'year', 'studios', 'rank']], on='anime_id')
    hybrid_df = hybrid_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

    return hybrid_df

# Generate hybrid recommendations for the new user
hybrid_recs_new_user = hybrid_recommendations_for_new_user(
    new_user_ratings=new_user_ratings,
    svd_model=svd,
    cosine_sim=cosine_sim,
    anime_df=anime_filtered_df,
    top_n=30,
    cf_weight=0.1,
    content_weight=0.9
)

def diversify_recommendations_by_keyword(df, column='name', max_per_keyword=1):
    keyword_counts = {}
    diversified = []
    
    for _, row in df.iterrows():
        name = row[column]
        # Extract keywords 
        keywords = [word.lower() for word in name.split() if len(word) > 3]  # Keywords are words > 3 chars
        
        # Check if the current keywords exceed the threshold
        if all(keyword_counts.get(keyword, 0) < max_per_keyword for keyword in keywords):
            diversified.append(row)
            for keyword in keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

    return pd.DataFrame(diversified)

def diversify_recommendations_by_series_keyword(df, column='name', keywords=None, max_per_keyword=1):
    
    keyword_counts = {}
    diversified = []

    for _, row in df.iterrows():
        name = row[column].lower()
        keyword_found = None

        # Check for series-level keywords in the name
        if keywords:
            for keyword in keywords:
                if keyword.lower() in name:
                    keyword_found = keyword.lower()
                    break

        # If no specific keyword found then allow entry
        if not keyword_found:
            diversified.append(row)
        else:
            # Limit recommendations for a specific keyword
            if keyword_counts.get(keyword_found, 0) < max_per_keyword:
                diversified.append(row)
                keyword_counts[keyword_found] = keyword_counts.get(keyword_found, 0) + 1

    return pd.DataFrame(diversified)


# Extract 'name' column
anime_titles = anime_filtered_df['name']

keywords = []
for title in anime_titles:
    # Split on spaces or punctuation
    keywords.extend(title.lower().split())

keyword_counts = Counter(keywords)

# Filter to include only keywords that occur more than once
series_keywords = [keyword for keyword, count in keyword_counts.items() if count > 3]

# Diversify recommendations based on series keywords
diversified_recs = diversify_recommendations_by_series_keyword(
    df=hybrid_recs_new_user,
    column='name',
    keywords=series_keywords,
    max_per_keyword=2 
)

