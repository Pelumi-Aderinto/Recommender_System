import numpy as np
import pandas as pd

def search_movies(df, search_phrase):
    search_phrase = search_phrase.lower()
    filtered_df = df[df['title'].str.lower().str.contains(search_phrase)]
    return filtered_df[['title', 'movieId', 'movie_index']]

def get_top_predictions(user_latent_vectors, item_latent_vectors, item_biases, movie_data_list, movies_df, top_n=20, min_users_ratings=300):
    movie_scores = np.einsum('i,ji->j', user_latent_vectors, item_latent_vectors) + 0.01 * item_biases
    movie_indices = np.argsort(movie_scores)[::-1]
    top_predictions = []

    for index in movie_indices:
        if len(movie_data_list[index]) >= min_users_ratings:
            top_predictions.append(movies_df.loc[movies_df['movie_index'] == index, 'title'].iloc[0])
            if len(top_predictions) >= top_n:
                break

    return top_predictions
