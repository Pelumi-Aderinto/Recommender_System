# MovieLens Analysis

# MovieLens Analysis

This project is a MovieLens dataset analysis and recommendation system using matrix factorization.

## Project Structure

movielens_analysis/
├── data/
│   ├── __init__.py
│   ├── download.py
│   ├── process.py
├── models/
│   ├── __init__.py
│   ├── model.py
├── utils/
│   ├── __init__.py
│   ├── helper.py
├── __init__.py
├── main.py
├── README.md
└── setup.py

```
python main.py --task download
python main.py --task preprocess
python main.py --task train_biases
python main.py --task train_latent
python main.py --task plot_embeddings
python main.py --task train_new_user
python main.py --task search_movies
python main.py --task top_predictions
python main.py --task train_feature_embeddings
```
