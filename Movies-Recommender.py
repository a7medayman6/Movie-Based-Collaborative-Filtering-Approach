import os

import sys 

import streamlit as st

import pandas as pd

import pickle

from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors

from streamlit import cli as stcli

def Preprocessing(root):
    # Read the data
    movies_df = pd.read_csv(os.path.join(root, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(root, "ratings.csv"))

    # PREPROCESSING
    data = ratings_df.pivot(index='movieId',columns='userId',values='rating')
    data.fillna(0,inplace=True)

    no_user_voted = ratings_df.groupby('movieId')['rating'].agg('count')
    no_movies_voted = ratings_df.groupby('userId')['rating'].agg('count')

    data = data.loc[no_user_voted[no_user_voted > 10].index,:]
    data = data.loc[:,no_movies_voted[no_movies_voted > 50].index]

    # create the matrix
    csr_data = csr_matrix(data.values)
    data.reset_index(inplace=True)

    return data, csr_data, movies_df, ratings_df

def LoadKNNModel(model_path, k):
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k, n_jobs=-1)
    model_file_r = open(model_path, 'rb')
    model = pickle.load(model_file_r)
    model_file_r.close()

    return model

def Recommend(model, dataset, csr_data, movies_df, movieName, n):
    """
        Recommend:          Recommends the closes n movies from the output of a KNN model trained on the MovieLens dataset.
        movieName (str):    The movie name required to get the closest n movies to.
        n (int):            The number of movies to get.
    """

    n += 1

    # get all movies conatins the movieName
    movies = movies_df[movies_df['title'].str.contains(movieName)]  

    # if there isn't any movie with "movieName"
    if len(movies) < 1:
        return False, str(movieName) + " movie doesn't exist in our data"

    # get the index of the first result
    movie_id = movies.iloc[0]['movieId']
    movie_index = dataset[dataset['movieId'] == movie_id].index[0]

    # get the indices of the closest n neighbors from the results on the knn model
    # the indices are the movies positions in the dataset
    distances, indices = model.kneighbors(csr_data[movie_index], n_neighbors=n)    

    # sort the output of the knn
    closest_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
    
    movies_to_recommend = []
    
    # get the movie name and add it to a list 
    for movie_prediction in reversed(closest_movie_indices):
        index = movie_prediction[0]
        distance = movie_prediction[1]
        
        # get the movie index
        movie_id = dataset.iloc[index]['movieId']
        movie_index = movies_df[movies_df['movieId'] == movie_id].index
        
        # get the movie name by movie index
        movie_name = movies_df.iloc[movie_index]['title'].values[0]
        
        # add movie name and distance from the original movie to the list
        movies_to_recommend.append({'Movie Name': movie_name,
                                    'Distance': distance})
       
    # create a data frame from the list
    df = pd.DataFrame(movies_to_recommend, index=range(1,n))

    return True, df


def main():

    # CONSTANTS
    dataset_root = os.path.join('.', 'ml-latest-small')

    model_dir = os.path.join('.', 'Models')
    model_name = 'movies-recommender.knn.pkl'
    model_path = os.path.join(model_dir, model_name)

    K = 20
    number_of_movies = 20

    # GET THE PREPROCESSED DATA
    data, csr_data, movies_df, ratings_df = Preprocessing(dataset_root)

    # LOAD THE MODEL
    model = LoadKNNModel(model_path, K)

    st.title('Movies Recommender')
    st.markdown('''##### Movies Recommender recommends **20** simmilar movies using Item Based Collaborative Filtering Approach by a trained K-Nearest-Neighbors Model on the MoviesLens dataset.''')
    movie_name = str(st.text_input('Movie Name')).title()
    number_of_movies = st.number_input('Number of Generated Movies', min_value=1, max_value=50, value=10, step=1)

    if st.button('Recommend Simmilar Movies'):
        success, recommended_movies = Recommend(model, data, csr_data, movies_df, movie_name, number_of_movies)
        print("Movie Name is ", movie_name)
        if success:
            st.dataframe(recommended_movies)
        else:
            st.markdown('##### ' + recommended_movies)



if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())