import os

import sys 

import streamlit as st

import pandas as pd

import pickle

import random

import string

from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors

from streamlit import cli as stcli

import matplotlib.pyplot as plt


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
        return str(movieName) + " movie doesn't exist in our data"

    # get the index of the first result
    movie_id = movies.iloc[0]['movieId']
    movie_index = dataset[dataset['movieId'] == movie_id].index[0]
    
    # get the name of the target movie
    target_movie = movies_df[movies_df['title'].str.contains(movieName)].iloc[0]
    target_movie = target_movie['title']
    
    
    # get the indices of the closest n neighbors from the results of the knn model
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
    df = pd.DataFrame(movies_to_recommend)

    return True, df, distances, indices, target_movie
   
def vis(x, y, data):
    fig = plt.figure(figsize = (20, 10))
    plt.scatter(x, y)
    plt.xlabel("Indix")
    plt.ylabel("Distance")
    plt.scatter(x[0],y[0], c ="green", s=50)

    for i, txt in enumerate(data):
        plt.annotate(txt, (x[i], y[i]))
    for i in range(len(x)):
        x_, y_ = [x[0], x[i]], [y[0], y[i]]
        plt.plot(x_, y_)
    
    # plt.show()    

    return fig

def main():

    # CONSTANTS
    dataset_root = os.path.join('.', 'ml-latest-small')

    model_dir = os.path.join('.', 'Models')
    model_name = 'movies-recommender.knn.pkl'
    model_path = os.path.join(model_dir, model_name)

    K = 10
    number_of_movies = 20

    # GET THE PREPROCESSED DATA
    data, csr_data, movies_df, ratings_df = Preprocessing(dataset_root)

    # LOAD THE MODEL
    model = LoadKNNModel(model_path, K)

    st.title('Movies Recommender')
    st.markdown('''##### Movies Recommender recommends simmilar movies using Item Based Collaborative Filtering Approach by a trained K-Nearest-Neighbors Model on the MoviesLens dataset.''')
    movie_name = str(st.text_input('Movie Name', placeholder='Recommends random movies if this field is empty.')).title().strip()
    number_of_movies = st.number_input('Number of Generated Movies', min_value=1, max_value=50, value=10, step=1)

    empty_input = True if movie_name == '' else False
    if st.button('Recommend Simmilar Movies'):
  
        while movie_name == '':

            movie_name = str(random.choice(movies_df['title'].tolist())).strip()
            if '(' in movie_name:
                movie_name = movie_name.split("(")[0].split()[0]
                success, recommended_movies, distances, indicies, target_movie = Recommend(model, data, csr_data, movies_df, movie_name, number_of_movies)
                if not success:
                    movie_name = ''

        success, recommended_movies, distances, indicies, target_movie = Recommend(model, data, csr_data, movies_df, movie_name, number_of_movies)
        if success:
            if not empty_input:
                st.markdown("Recommended Movies Like : " + movie_name)
            else:
                st.markdown("Random Recommended Movies")
            st.dataframe(recommended_movies)
            x, y = indicies.reshape((indicies[0,:].shape[0])), distances.reshape((distances[0,:].shape[0]))
            dat = recommended_movies['Movie Name'].tolist()
            #dat = [item.split("(")[0].split()[0] for item in dat]
            dat.insert(0, target_movie)
            fig = vis(x, y, dat)
            st.pyplot(fig)
        else:
            st.markdown('##### ' + recommended_movies)



if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())