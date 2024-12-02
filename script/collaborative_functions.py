new_user_ratings = [
    (0, 1, 8),  # User ID 0, Anime ID 1, Rating 8 
    (0, 6, 7),  # User ID 0, Anime ID 6, Rating 9
    (0, 19, 10), # User ID 0, Anime ID 19, Rating 10
]

# List of all anime IDs
all_anime_ids = user_clean['anime_id'].unique()

# Anime already rated by the new user
rated_anime_ids = [rating[1] for rating in new_user_ratings]

# Generate predictions for all other anime
recommendations = []
for anime_id in all_anime_ids:
    if anime_id not in rated_anime_ids:
        # Predict for the new user (user_id = 0) and unseen anime
        pred = svd.predict(uid=0, iid=anime_id)
        recommendations.append((anime_id, pred.est))

# Sort recommendations by predicted rating
recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

# Top 10 recommendations
top_10 = recommendations[:10]
print("Top 10 Recommendations:")
for anime_id, rating in top_10:
    print(f"Anime ID: {anime_id}, Predicted Rating: {rating}")

# anime_filtered_df has columns 'anime_id', 'name', and 'genres'
top_10_df = pd.DataFrame(top_10, columns=['anime_id', 'predicted_rating'])
top_10_detailed = pd.merge(top_10_df, anime_filtered_df, on='anime_id')

# Display top 10 with details
top_10_detailed[['anime_id', 'name', 'genres', 'predicted_rating']]