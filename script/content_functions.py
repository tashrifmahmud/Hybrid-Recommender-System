# Preprocessing function

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join back into a single string
    return ' '.join(tokens)

# Function to get recommendations with anime_id

def get_recommendations(anime_id, cosine_sim=cosine_sim, df=anime, top_n=10):
    # Find index of the given anime_id
    idx = df.index[df['anime_id'] == anime_id][0]
    
    # Get similarity scores for this anime
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top_n similar anime
    top_anime_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    # Combine results into a DataFrame
    recommendations = df.iloc[top_anime_indices][['anime_id', 'name', 'genres']].copy()
    
    return recommendations

def get_recommendations_with_scores(anime_id, cosine_sim=cosine_sim, df=anime, top_n=10):
    # Find index of the given anime_id
    idx = df.index[df['anime_id'] == anime_id][0]
    
    # Get similarity scores for this anime
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top_n similar anime
    top_anime_indices = [i[0] for i in sim_scores[1:top_n+1]]
    top_sim_scores = [i[1] for i in sim_scores[1:top_n+1]]
    
    # Combine results into a DataFrame
    recommendations = df.iloc[top_anime_indices][['anime_id', 'name', 'genres']].copy()
    recommendations['similarity_score'] = top_sim_scores
    
    return recommendations

def get_recommendations_with_scores_and_sort(anime_id, cosine_sim=cosine_sim, df=anime, top_n=10):
    # Find index of the given anime_id
    idx = df.index[df['anime_id'] == anime_id][0]
    
    # Get similarity scores for this anime
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top_n similar anime indices
    top_anime_indices = [i[0] for i in sim_scores[1:top_n*2]] 
    top_sim_scores = [i[1] for i in sim_scores[1:top_n*2]]
    
    # Combine results into a DataFrame
    recommendations = df.iloc[top_anime_indices][['anime_id', 'name', 'genres', 'popularity']].copy()
    recommendations['similarity_score'] = top_sim_scores
    
    # Sort by similarity_score and popularity
    recommendations = recommendations.sort_values(by=['similarity_score', 'popularity'], ascending=[False, True])
    
    # Return the top_n sorted recommendations
    return recommendations.head(top_n)

def get_recommendations_with_filter_and_weight(anime_id, cosine_sim=cosine_sim, df=anime, top_n=10, popularity_threshold=10000):
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
    recommendations = df.iloc[top_anime_indices][['anime_id', 'name', 'genres', 'popularity']].copy()
    recommendations['similarity_score'] = top_sim_scores
    
    # Filter based on popularity threshold
    recommendations = recommendations[recommendations['popularity'] <= popularity_threshold]
    
    # Weight similarity_score and popularity
    recommendations['weighted_score'] = (
        0.7 * recommendations['similarity_score'] + 
        0.3 * (1 / (recommendations['popularity'] + 1))
    )
    
    # Sort by weighted_score
    recommendations = recommendations.sort_values(by='weighted_score', ascending=False)
    
    # Return top_n recommendations
    return recommendations.head(top_n)

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
    recommendations['score_norm'] = recommendations['score'] / 10  
    
    # Weighted score: Adjust weights to balance factors
    recommendations['weighted_score'] = (
        0.4 * recommendations['similarity_score'] +
        0.2 * recommendations['popularity_norm'] +
        0.2 * recommendations['score_norm'] +
        0.2 * recommendations['rank_norm']
    )
    
    # Sort by weighted_score
    recommendations = recommendations.sort_values(by='weighted_score', ascending=False)
    
    # Return top_n recommendations
    return recommendations.head(top_n)