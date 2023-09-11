# %%
#!pip install fuzzywuzzy

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import pandas as pd
print('Pandas version: ', pd.__version__)

import numpy as np
print('NumPy version: ', np.__version__)

import matplotlib
print('Matplotlib version: ', matplotlib.__version__)

from matplotlib import pyplot as plt

import sklearn
print('Scikit-Learn version: ', sklearn.__version__)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cluster import KMeans


import pickle
print('Pickle version: ', pickle.format_version)

import sys
print('Sys version: ', sys.version[0:5])

from sklearn.neighbors import NearestNeighbors

import random

# %%
base_path = "/content/drive/MyDrive/TCC/Datasets/"
base_path = "datasets/"
# %%
ratings = pd.read_csv(base_path + 'ratings.csv', usecols=['userId','movieId','rating'])
movies = pd.read_csv(base_path + 'movies.csv', usecols=['movieId','title'])
ratings2 = pd.merge(ratings, movies, how='inner', on='movieId')
movies2 = pd.read_csv(base_path + 'movies.csv')

# %%
ratings2.head()

# %%
df = ratings2.pivot_table(index='title',columns='userId',values='rating').fillna(0)
df1 = df.copy()

# %%
df1.head()

# %%
def create_missing_df(dataframe):

  missing_index = dataframe.columns.tolist()
  missing = dataframe.isnull().sum().tolist()
  missing_df = pd.DataFrame({'Missing':missing}, index=missing_index)

  return missing_df

# %%
create_missing_df(movies2)

# %%
# the function to extract titles
def extract_title(title):

  year = title[len(title)-5:len(title)-1]

  # some movies do not have the info about year in the column title. So, we should take care of the case as well.
  if year.isnumeric():
    title_no_year = title[:len(title)-7]
    return title_no_year

  else:
    return title

# %%
# the function to extract years
def extract_year(title):

  year = title[len(title)-5:len(title)-1]

  # some movies do not have the info about year in the column title. So, we should take care of the case as well.
  if year.isnumeric():
    return int(year)

  else:
    return np.nan

# %%
movies2.rename(columns={'title':'title_year'}, inplace=True) # change the column name from title to title_year
movies2['title_year'] = movies2['title_year'].apply(lambda x: x.strip()) # remove leading and ending whitespaces in title_year
movies2['title'] = movies2['title_year'].apply(extract_title) # create the column for title
movies2['year'] = movies2['title_year'].apply(extract_year) # create the column for year

# %%
create_missing_df(movies2)

# %%
r,c = movies2[movies2['genres']=='(no genres listed)'].shape
print('The number of movies which do not have info about genres:',r)

# %%
movies2 = movies2[~(movies2['genres']=='(no genres listed)')].reset_index(drop=True)

# %%
movies2[['title','genres']].head(5)

# %%


# %%
# remove '|' in the genres column
movies2['genres'] = movies2['genres'].str.replace('|',' ')

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
movies2['genres'] = movies2['genres'].str.replace('Sci-Fi','SciFi')
movies2['genres'] = movies2['genres'].str.replace('Film-Noir','Noir')

# %%
tfidf_vector = TfidfVectorizer(stop_words='english') # create an object for TfidfVectorizer
tfidf_matrix = tfidf_vector.fit_transform(movies2['genres']) # apply the object to the genres column

# %%
# the first row vector of tfidf_matrix (Toy Story)
tfidf_matrix.todense()[0]

# %%
from sklearn.metrics.pairwise import linear_kernel

# %%
sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix) # create the cosine similarity matrix
print(sim_matrix)

# %%
# the function to convert from index to title_year
def get_title_year_from_index(index):

  return movies2[movies2.index == index]['title_year'].values[0]

# the function to convert from title to index
def get_index_from_title(title):

  return movies2[movies2.title == title].index.values[0]

# %%
from fuzzywuzzy import fuzz

# %%
def matching_score(a,b):

  return fuzz.ratio(a,b)

# %%
# a function to convert index to title
def get_title_from_index(index):

  return movies2[movies2.index == index]['title'].values[0]

# %%
# the function to return the most similar title to the words a user types
def find_closest_title(title):

  leven_scores = list(enumerate(movies2['title'].apply(matching_score, b=title)))
  sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
  closest_title = get_title_from_index(sorted_leven_scores[0][0])
  distance_score = sorted_leven_scores[0][1]

  return closest_title, distance_score

# %%
def contents_based_recommender(movie_user_likes, how_many):

  closest_title, distance_score = find_closest_title(movie_user_likes)
  rec_movie = []
  if distance_score == 100:

    movie_index = get_index_from_title(closest_title)
    movie_list = list(enumerate(sim_matrix[int(movie_index)]))
    similar_movies = list(filter(lambda x:x[0] != int(movie_index), sorted(movie_list,key=lambda x:x[1], reverse=True))) # remove the typed movie itself

    # print('Here\'s the list of movies similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')

    for i,s in similar_movies[:how_many]:
      rec_movie.append(get_title_year_from_index(i))

    return rec_movie

  else:
    # print('Did you mean '+'\033[1m'+str(closest_title)+'\033[0m'+'?','\n')

    movie_index = get_index_from_title(closest_title)
    movie_list = list(enumerate(sim_matrix[int(movie_index)]))
    similar_movies = list(filter(lambda x:x[0] != int(movie_index), sorted(movie_list,key=lambda x:x[1], reverse=True)))


  for i,s in similar_movies[:how_many]:
    rec_movie.append(get_title_year_from_index(i))

  return rec_movie


# %%
def recommend_movies_cf(user, num_recommended_movies):
  recommended_movies = []

  for m in df[df[user] == 0].index.tolist():

    index_df = df.index.tolist().index(m)
    predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
    recommended_movies.append((m, predicted_rating))

  sorted_rm = sorted(recommended_movies, key=lambda x:x[1], reverse=True)
  rank = 1
  return sorted_rm


# %%
def movie_recommender_cf(user, num_neighbors, num_recommendation):

  number_neighbors = num_neighbors

  knn = NearestNeighbors(metric='cosine', algorithm='brute')
  knn.fit(df.values)
  distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)

  user_index = df.columns.tolist().index(user)

  for m,t in list(enumerate(df.index)):
    if df.iloc[m, user_index] == 0:
      sim_movies = indices[m].tolist()
      movie_distances = distances[m].tolist()

      if m in sim_movies:
        id_movie = sim_movies.index(m)
        sim_movies.remove(m)
        movie_distances.pop(id_movie)

      else:
        sim_movies = sim_movies[:num_neighbors-1]
        movie_distances = movie_distances[:num_neighbors-1]

      movie_similarity = [1-x for x in movie_distances]
      movie_similarity_copy = movie_similarity.copy()
      nominator = 0

      for s in range(0, len(movie_similarity)):
        if df.iloc[sim_movies[s], user_index] == 0:
          if len(movie_similarity_copy) == (number_neighbors - 1):
            movie_similarity_copy.pop(s)

          else:
            movie_similarity_copy.pop(s-(len(movie_similarity)-len(movie_similarity_copy)))

        else:
          nominator = nominator + movie_similarity[s]*df.iloc[sim_movies[s],user_index]

      if len(movie_similarity_copy) > 0:
        if sum(movie_similarity_copy) > 0:
          predicted_r = nominator/sum(movie_similarity_copy)

        else:
          predicted_r = 0

      else:
        predicted_r = 0

      df1.iloc[m,user_index] = predicted_r
  return recommend_movies_cf(user,num_recommendation)


# %%
b=(movie_recommender_cf(1, 30, 10))

# %%
b[0]

# %%
b[0][0]

# %%
teste=[]
for i in b:
  if i[1]>=5.0: #treshold value
    teste.append(i[0])
  else:
    break

# %%
a=df[1].sort_values(ascending=False)

# %%
test =[]
for i in range(len(a)):
  if a[i]>=5.0:
    test.append(a.index.tolist()[i])
  else:
    break

# %%
user_rec = []
for i in range(len(test)):
  user_rec.append(contents_based_recommender(test[i],10))

# %%
flat_list = [item for sublist in user_rec for item in sublist]

# %%
len(flat_list)

# %%
len(teste)

# %%
uniqueList = []
duplicateList = []

for i in flat_list:
    if i not in uniqueList:
        uniqueList.append(i)
    elif i not in duplicateList:
        duplicateList.append(i)

print(duplicateList)

# %%
print(set(teste).intersection(flat_list))

# %%
def hybrid_mode(user, treshold_value):
  user_cf=movie_recommender_cf(user, 10, 10)
  rec_cf=[]
  for i in user_cf:
    if i[1]>=treshold_value: #treshold value
      rec_cf.append(i[0])
    else:
      break

  user_reviews = df[user].sort_values(ascending=False)
  user_max =[]
  for i in range(len(user_reviews)):
    if a[i]>=5.0:
      user_max.append(user_reviews.index.tolist()[i])
    else:
      break
  rec_cb = []
  for i in range(len(user_max)):
    rec_cb.append(contents_based_recommender(user_max[i],10))
  rec_cb = [item for sublist in user_rec for item in sublist]

  rec_movies = set(rec_cf).intersection(rec_cb)

  return rec_movies

# %%


def lambda_handler(event, context):
    resp = hybrid_mode(user=12,treshold_value=5)
    return {
        'statusCode': 200,
        'body': resp
    }

