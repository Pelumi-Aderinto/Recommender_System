import pandas as pd
import numpy as np

def load_data():
    movies_df = pd.read_csv('ml-25m/movies.csv')
    ratings_df = pd.read_csv('ml-25m/ratings.csv')
    ratings_df = ratings_df.drop('timestamp', axis=1)
    return movies_df, ratings_df

def get_unique_genres(movies_df):
    unique_genres = set(movies_df['genres'].apply(lambda x: x.split('|')).explode())
    return unique_genres

def encode_genres(movies_df, unique_genres):
    genre_map = {genre: idx for idx, genre in enumerate(unique_genres)}
    movies_df['features'] = movies_df['genres'].apply(lambda x: [1 if genre in x.split('|') else 0 for genre in unique_genres])
    return movies_df, genre_map

def preprocess_data(movies_df, ratings_df):
    unique_genres = get_unique_genres(movies_df)
    movies_df, genre_map = encode_genres(movies_df, unique_genres)
    
    movies = ratings_df["movieId"].unique().tolist()
    users = ratings_df["userId"].unique().tolist()
    
    movieid_to_idx = {movie: idx for idx, movie in enumerate(movies)}
    userid_to_idx = {user: idx for idx, user in enumerate(users)}
    
    ratings_df['user_index'] = ratings_df['userId'].map(userid_to_idx)
    ratings_df['movie_index'] = ratings_df['movieId'].map(movieid_to_idx)
    
    movies_df['movie_index'] = movies_df['movieId'].map(movieid_to_idx)
    movies_df = movies_df.dropna(subset=['movie_index'])
    movies_df['movie_index'] = movies_df['movie_index'].astype(int)
    
    index_to_features = {row['movie_index']: row['features'] for _, row in movies_df.iterrows()}
    
    max_index = max(index_to_features.keys())
    feature_vectors = np.zeros((max_index + 1, len(genre_map)))
    for idx, features in index_to_features.items():
        feature_vectors[idx] = features
    
    return ratings_df, movies_df, feature_vectors, genre_map

def split_data(ratings_df, userid_to_idx, movieid_to_idx):
    key = np.random.default_rng(0)
    user_ids = ratings_df['userId'].tolist()
    movie_ids = ratings_df['movieId'].tolist()
    ratings = ratings_df['rating'].tolist()
    train_user_data_list = [[] for _ in range(ratings_df["userId"].nunique())]
    test_user_data_list = [[] for _ in range(ratings_df["userId"].nunique())]
    train_movie_data_list = [[] for _ in range(ratings_df["movieId"].nunique())]
    test_movie_data_list = [[] for _ in range(ratings_df["movieId"].nunique())]

    for user_id, movie_id, rating in zip(user_ids, movie_ids, ratings):
        if key.uniform(0, 1) < 0.8:
            train_user_data_list[userid_to_idx[user_id]].append((movieid_to_idx[movie_id], rating))
            train_movie_data_list[movieid_to_idx[movie_id]].append((userid_to_idx[user_id], rating))
        else:
            test_user_data_list[userid_to_idx[user_id]].append((movieid_to_idx[movie_id], rating))
            test_movie_data_list[movieid_to_idx[movie_id]].append((userid_to_idx[user_id], rating))

    train_user_data_list = [np.array(sublist) for sublist in train_user_data_list]
    test_user_data_list = [np.array(sublist) for sublist in test_user_data_list]
    train_movie_data_list = [np.array(sublist) for sublist in train_movie_data_list]
    test_movie_data_list = [np.array(sublist) for sublist in test_movie_data_list]
    
    return train_user_data_list, test_user_data_list, train_movie_data_list, test_movie_data_list

def all_data(ratings_df, userid_to_idx, movieid_to_idx):
    key = np.random.default_rng(0)
    user_ids = ratings_df['userId'].tolist()
    movie_ids = ratings_df['movieId'].tolist()
    ratings = ratings_df['rating'].tolist()
    user_data_list = [[] for _ in range(ratings_df["userId"].nunique())]
    movie_data_list = [[] for _ in range(ratings_df["movieId"].nunique())]


    for user_id, movie_id, rating in zip(user_ids, movie_ids, ratings):
        user_data_list[userid_to_idx[user_id]].append((movieid_to_idx[movie_id], rating))
        movie_data_list[movieid_to_idx[movie_id]].append((userid_to_idx[user_id], rating))


    user_data_list = [np.array(sublist) for sublist in user_data_list]
    movie_data_list = [np.array(sublist) for sublist in movie_data_list]


    print("Length of user_data_list:", len(user_data_list))
    print("Length of user_data_list 0:", len(user_data_list[0]))

    print("Length of movie_data_list:", len(movie_data_list))
    print("Length of movie_data_list 0:", len(movie_data_list[0]))

    
    return user_data_list, user_data_list