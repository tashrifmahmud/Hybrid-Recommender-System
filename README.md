# Hybrid Recommender System
> Tashrif Mahmud | :bar_chart: [Streamlit App](https://hybrid-recommender-system.streamlit.app) | :film_strip: [YouTube Video](https://www.youtube.com/watch?v=VUjMEUpg5ec)
> Tech Stack: Python | Pandas | Matplotlib | Scikit-Learn | Surprise | NLP (Sentence Transformers) | API (Jikan API) | Streamlit | Kaggle

![Hybrid-Recommender-System](image/0_Banner.jpg)

## Project Overview
We developed a powerful recommendation system for anime enthusiasts, seamlessly blending collaborative and content-based filtering methods. The system delivers personalized recommendations enhanced by the Jikan API, featuring images, synopses, and hybrid scoring for an enriched user experience.

## Dataset
We primarily used [MyAnimeList](https://www.kaggle.com/datasets/dsfelix/animes-dataset-2023) dataset to build our recommendation model. We also tried other similar dataset from Kaggle but our selected dataset is the latest and recently updated therefore would give us the best result.

## Dataset Preprocessing
We have conducted several preprocessing on our dataset before training our models. It can be seen in [1_data_cleaning_part_1](https://github.com/tashrifmahmud/Hybrid-Recommender-System/blob/main/1_data_cleaning_part_1.ipynb) notebook for content based filtering model and [2_data_cleaning_part_2](https://github.com/tashrifmahmud/Hybrid-Recommender-System/blob/main/2_data_cleaning_part_2.ipynb) notebook for collaborative model. Some steps are continued in remaining notebooks.

### Dataset Preprocessing Steps:

- Accessing data: Imported and accessed dataset alongside primary inspection, used pandas dataframe for easier cleaning
- Data cleaning: missing, null and duplicates values, placeholder titles, 'unknown' details, as well as NSFW titles have be removed.
- Data transformation: Got 'year' of anime from 'aired' columns, rating is simplified for textual relavance. 
- Dropping columns: We removed unnecessary metadata columns that doesn't improve our model
- Sparsity check: Sparsity of anime titles and review counts is manageed to simulate real life data.

After transforming all textual data into 'combined_text' column for content-based filtering, we also used the following pre-processing:

- Cleaning Text: Removing punctuation, newline characters and trailing spaces + Lowercasing characters
- Tokenization: Used NLTK word tokenizer for tokenizing words 
- Stop Word Removal: Filter out unnecessary words using nltk stopwords english corpus for own model

## Content-based filtering Model 1 using TFIDF

In our first attempt, we used TFID Vectorizer to generate embeddings for the cleaned tokens for our content-based model, process can be seen in [3_content_based_filtering_tfidf](https://github.com/tashrifmahmud/Hybrid-Recommender-System/blob/main/3_content_based_filtering_tfidf.ipynb) notebook. We also used standard scaler to scale our numerical data like 'year' and used hstack to combine our numerical and text data. Finally, we used cosine similarity method to complete our first model. The results are not good with this model so we took a different approach.

## Content-based Filtering Model 2 using Sentence Transformers 
We then used a pre-trained Sentence Transformer model, [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from Huggingface, to convert the combined_text into dense embeddings. Details can be seen in [4_content_based_filtering_st](https://github.com/tashrifmahmud/Hybrid-Recommender-System/blob/main/4_content_based_filtering_st.ipynb) notebook. Again, we scaled year to a comparable range (using Min-Max scaling) and combined the dense embeddings from Sentence Transformers with the scaled year feature. We again computed cosine similarity with the final features to complete our model.

### TFIDF vs Pre-trained Models Comparison:

![TFIDF vs Sentence Transformer](image/1_TFIDF_ST_model.png "Content-based Filtering: TFIDF vs Sentence Transformer")
As we can see here, the results are significantly better now. We will proceed with pre-trained model for content-based filtering approach.

## Collaborative Filtering Model
We used SVD (Singular Value Decomposition) which is efficient for decomposing large user-item matrices into latent factors. Users and animes are represented in a shared latent space, enabling similarity computation. Details can be seen in [5_collaborative_filtering](https://github.com/tashrifmahmud/Hybrid-Recommender-System/blob/main/5_collaborative_filtering.ipynb) notebook. We also split our data into train and test sets to validate our model and ensure there is no overfitting. 

### Hyperparamer Tuning:
We used Gridsearch CV for to fine-tune our SVD model. We settled with 200 number of latent factors(any more would result in overfitting the model), 20 number of epochs, 0.005 learning rate and 0.05 regularization. With this we get the best RMSE of 1.12 in the train set and best RMSE of 1.11 in the training set.

This was done in combination of manipulating the sparsity of number anime titles with number of ratings. Which further improved our model performance.

![Default vs Fine-Tuned](image/2_Collaborative_model.png "Collaborative Filtering: Default vs Fine-Tuned")
The fine-tuned model performs miles better and we will proceed with this approach.

## Hybrid Recommendation Model
The hybrid recommendation model combines the content-based model with collaborative model. 

- Hybrid Weight: Content | 60:40 | Collaborative. By experimenting with various weight ratio, we settled with 60% to content and 40% to collaborative filtering. We have balanced it through personalized insights with broader patterns using domain knowledge. 

- Diversify Function: To ensure that same titles with multiple sequels or movies are not repeated due to their similarity, we introduced diversity mechanism to limit it to 2 per recommendation.

- Popularity Mechanism: We also introduced 10% popularity bias to content-based model to ensure that trending and popular titles are shown to the user instead of just similar anime. This ensures latest and most watched series to get a small bump in the recommendation system. 

![Hybrid model](image/3_Hybrid_model.png "Hybrid Recommendation Results")
We get a decent recommendation from this model as we can see in the results, cross-checking anime platforms like MyAnimeList and Reddit forums strongly valiadates the output.

## App Deployment
We deployed a trimmed down version of our Hybrid Recommendation Model using [Streamlit App](https://hybrid-recommender-system.streamlit.app/). This version performs virtually the same. We also used Jikan API in our deployment version. The Streamlit App lets the user choose any 5 anime and rate them to their liking. Then it shows a summary of 10 most similar anime summary at the beginning, and as the user scrolls down they can find upto 40 most similar titles with more details generated from Jikan API.

### Jikan API Integration
The [Jikan API](https://jikan.moe/) is an Jikan unofficial & open-source API for the most active online anime community and database - MyAnimeList. We used this API to fetch details like titles, images, synopses, score, studios etc. to make our results look more complete. The user can simply click the link of each recommended anime titles and can be re-directed to the MyAnimeList website of it.

## Links
:bar_chart: [Streamlit App](https://hybrid-recommender-system.streamlit.app) | :film_strip: [YouTube Video](https://www.youtube.com/watch?v=VUjMEUpg5ec)