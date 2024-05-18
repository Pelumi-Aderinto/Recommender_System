import argparse
from data.download import download_and_unzip_data
from data.process import load_data, preprocess_data, split_data, all_data
from models.model import train_biases_only_model, train_latent_vectors_model, plot_embeddings, train_new_user, train_with_feature_embeddings
from utils.helper import search_movies, get_top_predictions

def main():
    parser = argparse.ArgumentParser(description='MovieLens Analysis')
    parser.add_argument('--task', type=str, choices=['download', 'preprocess', 'train_biases', 'train_latent', 'plot_embeddings', 'train_new_user', 'search_movies', 'top_predictions', 'train_feature_embeddings'], required=True, help='Task to run')
    args = parser.parse_args()

    if args.task == 'download':
        data_url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        output_zip = "moviedataset.zip"
        download_and_unzip_data(data_url, output_zip)

    elif args.task == 'preprocess':
        movies_df, ratings_df = load_data()
        ratings_df, movies_df, feature_vectors, genre_map = preprocess_data(movies_df, ratings_df)
        print("Data preprocessed successfully")

    elif args.task == 'train_biases':
        movies_df, ratings_df = load_data()
        ratings_df, movies_df, feature_vectors, genre_map = preprocess_data(movies_df, ratings_df)
        userid_to_idx = {user: idx for idx, user in enumerate(ratings_df["userId"].unique().tolist())}
        movieid_to_idx = {movie: idx for idx, movie in enumerate(ratings_df["movieId"].unique().tolist())}
        user_data_list, movie_data_list = all_data(ratings_df, userid_to_idx, movieid_to_idx)
        train_biases_only_model(user_data_list, movie_data_list)

    elif args.task == 'train_latent':
        movies_df, ratings_df = load_data()
        ratings_df, movies_df, feature_vectors, genre_map = preprocess_data(movies_df, ratings_df)
        userid_to_idx = {user: idx for idx, user in enumerate(ratings_df["userId"].unique().tolist())}
        movieid_to_idx = {movie: idx for idx, movie in enumerate(ratings_df["movieId"].unique().tolist())}
        train_user_data_list, _, train_movie_data_list, _ = split_data(ratings_df, userid_to_idx, movieid_to_idx)
        train_latent_vectors_model(train_user_data_list, train_movie_data_list, k=10)

    elif args.task == 'plot_embeddings':
        movies_df, ratings_df = load_data()
        ratings_df, movies_df, feature_vectors, genre_map = preprocess_data(movies_df, ratings_df)
        userid_to_idx = {user: idx for idx, user in enumerate(ratings_df["userId"].unique().tolist())}
        movieid_to_idx = {movie: idx for idx, movie in enumerate(ratings_df["movieId"].unique().tolist())}
        train_user_data_list, _, train_movie_data_list, _ = split_data(ratings_df, userid_to_idx, movieid_to_idx)
        _, _, _, item_latent_vectors = train_latent_vectors_model(train_user_data_list, train_movie_data_list, k=2)
        genre_indexes = {genre: [row['movie_index'] for _, row in movies_df.iterrows() if genre in row['genres'].split('|')] for genre in genre_map}
        plot_embeddings(item_latent_vectors, genre_indexes, movies_df)

    elif args.task == 'train_new_user':
        movies_df, ratings_df = load_data()
        ratings_df, movies_df, feature_vectors, genre_map = preprocess_data(movies_df, ratings_df)
        userid_to_idx = {user: idx for idx, user in enumerate(ratings_df["userId"].unique().tolist())}
        movieid_to_idx = {movie: idx for idx, movie in enumerate(ratings_df["movieId"].unique().tolist())}
        train_user_data_list, _, train_movie_data_list, _ = split_data(ratings_df, userid_to_idx, movieid_to_idx)
        _, _, _, item_latent_vectors = train_latent_vectors_model(train_user_data_list, train_movie_data_list, k=10)
        new_user = [(2189, 5)]
        user_latent_vectors = train_new_user(new_user, item_latent_vectors, item_biases)
        print(user_latent_vectors)

    elif args.task == 'search_movies':
        movies_df, _ = load_data()
        search_phrase = input('Enter movie search phrase: ')
        search_results = search_movies(movies_df, search_phrase)
        print(search_results)

    elif args.task == 'top_predictions':
        movies_df, ratings_df = load_data()
        ratings_df, movies_df, feature_vectors, genre_map = preprocess_data(movies_df, ratings_df)
        userid_to_idx = {user: idx for idx, user in enumerate(ratings_df["userId"].unique().tolist())}
        movieid_to_idx = {movie: idx for idx, movie in enumerate(ratings_df["movieId"].unique().tolist())}
        train_user_data_list, _, train_movie_data_list, _ = split_data(ratings_df, userid_to_idx, movieid_to_idx)
        _, _, _, item_latent_vectors = train_latent_vectors_model(train_user_data_list, train_movie_data_list, k=10)
        new_user = [(2189, 5)]
        user_latent_vectors = train_new_user(new_user, item_latent_vectors, item_biases)
        top_predictions = get_top_predictions(user_latent_vectors, item_latent_vectors, item_biases, train_movie_data_list, movies_df)
        print(top_predictions)

    elif args.task == 'train_feature_embeddings':
        movies_df, ratings_df = load_data()
        ratings_df, movies_df, feature_vectors, genre_map = preprocess_data(movies_df, ratings_df)
        userid_to_idx = {user: idx for idx, user in enumerate(ratings_df["userId"].unique().tolist())}
        movieid_to_idx = {movie: idx for idx, movie in enumerate(ratings_df["movieId"].unique().tolist())}
        train_user_data_list, test_user_data_list, train_movie_data_list, test_movie_data_list = split_data(ratings_df, userid_to_idx, movieid_to_idx)
        num_features = len(feature_vectors[0])
        genres = list(genre_map.keys())
        train_with_feature_embeddings(train_user_data_list, test_user_data_list, train_movie_data_list, test_movie_data_list, feature_vectors, num_features, genres)

if __name__ == "__main__":
    main()
