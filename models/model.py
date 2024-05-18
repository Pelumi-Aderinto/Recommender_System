import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_biases_only_model(user_data_list, movie_data_list, iter=100, lbda=1, gamma=0.0001):
    M = len(user_data_list)
    N = len(movie_data_list)
    user_biases = np.zeros((M))
    item_biases = np.zeros((N))
    losses = []
    errors = []
    print("we made it here!")

    for i in tqdm(range(iter)):
        for m in range(M):
            if len(user_data_list[m]) > 0:
                ratings = user_data_list[m][:, 1]
                indices = user_data_list[m][:, 0].astype(int)

                # Calculate user bias
                user_bias = lbda * np.sum(ratings - item_biases[indices]) / (lbda * len(indices) + gamma)
                user_biases[m] = user_bias

        for q in range(N):
            if len(movie_data_list) > 0:
                ratings = movie_data_list[q][:, 1]
                indices = movie_data_list[q][:, 0].astype(int)
                # Calculate user bias
                item_bias = lbda * np.sum(ratings - user_biases[indices]) / (lbda * len(indices) + gamma)
                item_biases[q] = item_bias

        # Calculate training loss and RMSE
        train_error_squared = 0
        train_size = 0

        for m in range(M):
            if len(user_data_list[m]) > 0:
                train_ratings = user_data_list[m][:, 1]
                train_indices = user_data_list[m][:, 0].astype(int)
                train_error_squared += np.sum((train_ratings - user_biases[m] - item_biases[train_indices])**2)
                train_size += len(train_indices)

        train_loss = -0.5 * lbda * train_error_squared - 0.5 * gamma * (np.sum(user_biases**2) + np.sum(item_biases**2))
        train_error = np.sqrt(1 / train_size * train_error_squared)

        losses.append(-train_loss)
        errors.append(train_error)

    plt.plot(losses)
    plt.xlabel('Number of iterations')
    plt.ylabel('Negative loss')
    plt.savefig('Neg_regularized_log_likelihood_loss1.pdf', bbox_inches='tight')

    plt.plot(errors)
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.savefig('Neg_regularized_log_likelihood_rmse1.pdf', bbox_inches='tight')

    return user_biases, item_biases

def train_latent_vectors_model(user_data_list, movie_data_list, k, iter=100, lbda=1, gamma=0.001, tau=1):
    M = len(user_data_list)
    N = len(movie_data_list)
    user_biases = np.zeros((M))
    item_biases = np.zeros((N))
    user_latent_vectors = np.random.normal(loc=0, scale=1/np.sqrt(k), size=(M, k))
    item_latent_vectors = np.random.normal(loc=0, scale=1/np.sqrt(k), size=(N, k))
    losses = []
    errors = []

    for i in tqdm(range(iter)):
        for m in range(M):
            if len(user_data_list[m]) > 0:
                ratings = user_data_list[m][:, 1]
                indices = user_data_list[m][:, 0].astype(int)
                user_bias = lbda * np.sum(ratings - item_biases[indices]) / (lbda * len(indices) + gamma)
                user_biases[m] = user_bias

                left = np.sum(np.einsum('ij,il->ijl', item_latent_vectors[indices], item_latent_vectors[indices]), axis=0)
                right = np.sum(np.einsum('ji,j->ji', item_latent_vectors[indices], ratings - user_bias - item_biases[indices]), axis=0)
                user_latent_vectors[m] = np.linalg.solve(lbda * left + tau * np.eye(k), lbda * right)

        for q in range(N):
            if len(movie_data_list[q]) > 0:
                ratings = movie_data_list[q][:, 1]
                indices = movie_data_list[q][:, 0].astype(int)
                item_bias = lbda * np.sum(ratings - user_biases[indices]) / (lbda * len(indices) + gamma)
                item_biases[q] = item_bias

                left = np.sum(np.einsum('ij,il->ijl', user_latent_vectors[indices], user_latent_vectors[indices]), axis=0)
                right = np.sum(np.einsum('ji,j->ji', user_latent_vectors[indices], ratings - user_biases[indices] - item_bias), axis=0)
                item_latent_vectors[q] = np.linalg.solve(lbda * left + tau * np.eye(k), lbda * right)

        error_squared = 0
        train_size = 0

        for m in range(M):
            if len(user_data_list[m]) > 0:
                ratings = user_data_list[m][:, 1]
                indices = user_data_list[m][:, 0].astype(int)
                error_squared += np.sum((ratings - np.einsum('i,ji->j', user_latent_vectors[m], item_latent_vectors[indices]) - user_biases[m] - item_biases[indices])**2)
                train_size += len(indices)

        loss = -0.5 * lbda * error_squared - 0.5 * gamma * np.sum(user_biases**2) - 0.5 * gamma * np.sum(item_biases**2) - 0.5 * tau * (np.einsum('ij,ij->', item_latent_vectors, item_latent_vectors) + np.einsum('ij,ij->', user_latent_vectors, user_latent_vectors))
        error = np.sqrt(1 / train_size * error_squared)

        losses.append(-loss)
        errors.append(error)

    plt.plot(errors)
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.legend(['k=' + str(k)])
    plt.savefig(f'RMSE_for_k_{k}.pdf', bbox_inches='tight')

    plt.plot(losses)
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.legend(['k=' + str(k)])
    plt.savefig(f'Loss_for_k_{k}.pdf', bbox_inches='tight')

    return user_biases, item_biases, user_latent_vectors, item_latent_vectors

def plot_embeddings(item_latent_vectors, genre_indexes, movies_df):
    plt.figure(figsize=(10,8))
    plt.scatter(item_latent_vectors[:,0], item_latent_vectors[:,1], s=5, c='orange')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('movies_embeddings_1.pdf', bbox_inches='tight')

    popular_movies_index = [idx for idx, movie in enumerate(genre_indexes) if len(movie) > 300]
    popular_movies_embedding = item_latent_vectors[popular_movies_index]

    plt.figure(figsize=(10,8))
    plt.scatter(popular_movies_embedding[:,0], popular_movies_embedding[:,1], s=5, c='orange')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('movies_embeddings_2.pdf', bbox_inches='tight')

    emb1 = item_latent_vectors[genre_indexes['Children']]
    emb2 = item_latent_vectors[genre_indexes['Horror']]

    plt.figure(figsize=(10,8))
    plt.scatter(emb1[:,0], emb1[:,1], s=5, label='Children', c='green')
    plt.scatter(emb2[:,0], emb2[:,1], s=5, label='Horror', c='red')
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('movies_embeddings_3.pdf', bbox_inches='tight')

def train_new_user(user_list, item_latent_vectors, item_biases, k=10, num_iter=100, lbda=0.01, gamma=0.001, tau=0.001):
    user_latent_vectors = np.zeros(k)
    user_list = np.array(user_list)

    for i in range(num_iter):
        ratings = user_list[:, 1]
        indices = user_list[:, 0].astype(int)
        user_bias = lbda * np.sum(ratings - np.einsum('i,ji->j', user_latent_vectors, item_latent_vectors[indices]) - item_biases[indices]) / (lbda * len(indices) + gamma)

        left = np.sum(np.einsum('ij,il->ijl', item_latent_vectors[indices], item_latent_vectors[indices]), axis=0)
        right = np.sum(np.einsum('ji,j->ji', item_latent_vectors[indices], ratings - user_bias - item_biases[indices]), axis=0)
        user_latent_vectors = np.linalg.solve(lbda * left + tau * np.eye(k), lbda * right)

    return user_latent_vectors

def train_with_feature_embeddings(train_user_data_list, test_user_data_list, train_movie_data_list, test_movie_data_list, index_to_features, num_features, genres, iter=20, k=2, lbda=5, gamma=0.1, tau=1):
    M = len(train_user_data_list)
    N = len(train_movie_data_list)
    user_biases = np.zeros((M))
    item_biases = np.zeros((N))
    user_latent_vectors = np.random.normal(loc=0, scale=1/np.sqrt(k), size=(M, k))
    item_latent_vectors = np.random.normal(loc=0, scale=1/np.sqrt(k), size=(N, k))
    f_l = np.random.normal(loc=0, scale=1/np.sqrt(k), size=(num_features, k))
    train_errors = []
    test_errors = []

    for i in tqdm(range(iter)):
        plt.figure(figsize=(10,8))
        plt.scatter(f_l[:,0], f_l[:,1], s=100, c='orange')
        for j, genre in enumerate(genres):
            plt.text(f_l[j,0], f_l[j,1], genre, fontsize=12, ha='center', va='center')
        plt.title('2D Embeddings Plot with Features')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(f'feature_embeddings_iter_{i}.pdf', bbox_inches='tight')

        for m in range(M):
            if len(train_user_data_list[m]) > 0:
                ratings = train_user_data_list[m][:, 1]
                indices = train_user_data_list[m][:, 0].astype(int)
                user_bias = lbda * np.sum(ratings - np.einsum('i,ji->j', user_latent_vectors[m], item_latent_vectors[indices]) - item_biases[indices]) / (lbda * len(indices) + gamma)
                user_biases[m] = user_bias

                left = np.sum(np.einsum('ij,il->ijl', item_latent_vectors[indices], item_latent_vectors[indices]), axis=0)
                right = np.sum(np.einsum('ji,j->ji', item_latent_vectors[indices], ratings - user_bias - item_biases[indices]), axis=0)
                user_latent_vectors[m] = np.linalg.solve(lbda * left + tau * np.eye(k), lbda * right)

        for n in range(N):
            if len(train_movie_data_list[n]) > 0:
                ratings = train_movie_data_list[n][:, 1]
                indices = train_movie_data_list[n][:, 0].astype(int)
                item_bias = lbda * np.sum(ratings - np.einsum('ij,j->i', user_latent_vectors[indices], item_latent_vectors[n]) - user_biases[indices]) / (lbda * len(indices) + gamma)
                item_biases[n] = item_bias

                left = np.sum(np.einsum('ij,il->ijl', user_latent_vectors[indices], user_latent_vectors[indices]), axis=0)
                right = np.sum(np.einsum('ji,j->ji', user_latent_vectors[indices], ratings - user_biases[indices] - item_bias), axis=0)
                F_n = np.sum(index_to_features[n])
                item_latent_vectors[n] = np.linalg.solve(left + tau * np.eye(k), right + tau * (np.dot(index_to_features[n], f_l[:]) / np.sqrt(F_n)))

        for i in range(num_features):
            indices = np.where(index_to_features[:, i] == 1)
            feature_sums = np.sum(index_to_features[indices], axis=1)
            features_except_i = np.delete(index_to_features, i, axis=1)[indices]
            F_except_i = np.delete(f_l, i, axis=0)

            total_right = np.sum(np.einsum('ij, i -> ij', item_latent_vectors[indices], 1/np.sqrt(feature_sums))
                                 - np.einsum('i, ij-> ij', 1/feature_sums , np.einsum('ij, jk -> ik', features_except_i, F_except_i)), axis=0)
            f_l[i] = total_right / (np.sum(1/np.sqrt(feature_sums)) - 1)

        train_error_squared = 0
        train_size = 0
        test_error_squared = 0
        test_size = 0

        for m in range(M):
            if len(train_user_data_list[m]) > 0:
                train_ratings = train_user_data_list[m][:, 1]
                train_indices = train_user_data_list[m][:, 0].astype(int)
                train_error_squared += np.sum((train_ratings - np.einsum('i,ji->j', user_latent_vectors[m], item_latent_vectors[train_indices]) - user_biases[m] - item_biases[train_indices])**2)
                train_size += len(train_indices)

            if len(test_user_data_list[m]) > 0:
                test_ratings = test_user_data_list[m][:, 1]
                test_indices = test_user_data_list[m][:, 0].astype(int)
                test_error_squared += np.sum((test_ratings - np.einsum('i,ji->j', user_latent_vectors[m], item_latent_vectors[test_indices]) - user_biases[m] - item_biases[test_indices])**2)
                test_size += len(test_indices)

        train_error = np.sqrt(1 / train_size * train_error_squared)
        test_error = np.sqrt(1 / test_size * test_error_squared)

        train_errors.append(train_error)
        test_errors.append(test_error)

        print(f"Iteration {i}: train RMSE {train_error}, test RMSE {test_error}")

    return train_errors, test_errors
