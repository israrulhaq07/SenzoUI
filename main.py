

# import numpy as np
# import pandas as pd
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# def recommend_movies(genre, duration, era, rating):
#     # Load the movie metadata
#     movies = pd.read_csv('output1.csv')
#     movies['critically_acclaimed'] = movies['vote_average'].apply(lambda x: 'yes' if x > 7 else 'no')
#     # Filter the movies based on user input
#     #movies = movies[movies['genre_names'].str.contains(genre)]
#     movies = movies[movies['duration']== 'medium']
#     print(movies,"lllllllllllllll")
#     movies = movies[movies['era'].str.contains(era)]
#     movies = movies[movies['critically_acclaimed'].str.contains(rating)]

#     # Use SentenceTransformer to encode the movie texts and the user input
#     model = SentenceTransformer('bert-base-nli-mean-tokens')
#     movie_embeddings = model.encode(movies['overview'].to_list())
#     user_input_embedding = model.encode([genre])
#     #np.save('data.npy', movie_embeddings)
#     #movie_embeddings = np.load('data.npy')
#     # Calculate the cosine similarity between the user input and all movie texts
#     similarity_scores = cosine_similarity(user_input_embedding, movie_embeddings).flatten()

#     # Get the top 10 most similar movies
#     movie_titles = movies['original_title']
#     similar_movies_indices = similarity_scores.argsort()[:-11:-1]
#     similar_movies = movie_titles.iloc[similar_movies_indices]
#     similarities = similarity_scores[similar_movies_indices]

#     # Print the top 10 most similar movies
#     for i in range(len(similar_movies)):
#         print(f"Movie: {similar_movies.iloc[i]}, Similarity Score: {similarities[i]}")

#     return f"Top 10 movies similar to your input:\n{similar_movies}"


import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(user_input_genre, duration, era, rating):
    #movies = pd.read_csv('output1.csv')
    tmbd = pd.read_csv('output1.csv')

    df = pd.DataFrame(tmbd)
    df['critically_acclaimed'] = df['vote_average'].apply(lambda x: 'yes' if x > 7 else 'no')
    # filter rows based on 'city' column where value is 'SF'
    df = df[df['duration'].str.contains(duration)]
    df = df[df['era'].str.contains(era)]
    movies = df[df['critically_acclaimed'].str.contains(rating)]
    #print(typrmovies
    movies['text'] = movies[['genre_names']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # Combine user input into a single text string
    user_input_text = ' '.join(user_input_genre)

    # Use SentenceTransformer to encode the movie texts and the user input
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    movie_embeddings = model.encode(movies['text'].to_list())
    #movie_embeddings = np.load('data.npy')
    #np.save('data.npy', movie_embeddings)
    user_input_embedding = model.encode([user_input_text])

    # Calculate the cosine similarity between the user input and all movie texts
    similarity_scores = cosine_similarity(user_input_embedding, movie_embeddings).flatten()

    # Get the top 10 most similar movies
    movie_titles = movies['original_title']
    similar_movies_indices = similarity_scores.argsort()[:-11:-1]
    similar_movies = movie_titles.iloc[similar_movies_indices]
    print(type(similar_movies))
    similarities = similarity_scores[similar_movies_indices]
    re=''
    for i in range(len(similar_movies)):
        res=f"Movie: {similar_movies.iloc[i]}\n"
        re+=res +'\n' #, Similarity Score: {similarities[i]}")

    return re #f"Top 10 movies similar to your input:\n{similar_movies}"

# Set up the Streamlit app
st.title("Movie Recommender")

# Define the options for each input variable
genre_options = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'TV Movie', 'War', 'Western']
duration_options = ["short", "medium", "long"]
era_options = ["1970s", "1980s", "1990s", "2000s", "2010s"]
rating_options = ['yes', 'no']

# Add the Streamlit widgets for the input variables
genre = st.multiselect('Which genre do you prefer to watch?', genre_options)
duration = st.selectbox('Would you like to watch a short/medium/long movie?', duration_options)
era = st.selectbox('Which era should this movie belong to?', era_options)
rating = st.selectbox('Finally, would you like to watch a critically acclaimed movie?', rating_options)

# Add a button to run the recommender
if st.button('Submit'):
    st.write(recommend_movies(genre, duration, era, rating))
