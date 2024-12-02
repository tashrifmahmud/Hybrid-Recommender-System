def get_recommendations_by_id(anime_id, anime_df, cosine_sim, top_n=10):
    try:
        # Get the index of the anime that matches the anime id
        idx = anime_df[anime_df['anime_id'] == anime_id].index[0]
    except IndexError:
        raise ValueError(f"Anime ID {anime_id} not found in the dataset.")
    
    # Get the pairwise similarity scores for this anime
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the anime based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top N similar anime
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Retrieve the recommended anime
    recommendations = anime_df.iloc[top_indices][['anime_id', 'name', 'genres', 'year', 'studios']].copy()
    recommendations['similarity_score'] = [sim_scores[i][1] for i in range(1, top_n+1)]
    
    return recommendations