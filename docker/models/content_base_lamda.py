import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz
import requests

path_data = "https://github.com/MatheusNakai/Datasets/raw/main/"
movies = pd.read_csv(path_data + 'Datasets/movies_cb.csv')
movieDB = pd.read_csv(path_data + 'Datasets/movies_bin.csv')

tfidf_vector = TfidfVectorizer(stop_words='english') # create an object for TfidfVectorizer
tfidf_matrix = tfidf_vector.fit_transform(movies['genres']) # apply the object to the genres column

sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix) # create the cosine similarity matrix


# create a function to find the closest title
def matching_score(a,b):

  return fuzz.ratio(a,b)

# the function to convert from title to index
def get_index_from_title(title):

  return movies[movies.title == title].index.values[0]
  
# a function to convert index to title
def get_title_from_index(index):

  return movies[movies.index == index]['title'].values[0]
  
# the function to return the most similar title to the words a user types
def find_closest_title(title):

  leven_scores = list(enumerate(movies['title_year'].apply(matching_score, b=title)))
  sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
  closest_title = get_title_from_index(sorted_leven_scores[0][0])
  distance_score = sorted_leven_scores[0][1]

  return closest_title, distance_score
  
def contents_based_recommender(movie_user_likes, how_many):

  closest_title, distance_score = find_closest_title(movie_user_likes)
  rec_movie = []
  if distance_score == 100:

    movie_index = get_index_from_title(closest_title)
    movie_list = list(enumerate(sim_matrix[int(movie_index)]))
    similar_movies = list(filter(lambda x:x[0] != int(movie_index), sorted(movie_list,key=lambda x:x[1], reverse=True))) # remove the typed movie itself

    for i,s in similar_movies[:how_many]:
      rec_movie.append(movies.iloc[i]['title'])

    return rec_movie

  else:
    # print('Did you mean '+'\033[1m'+str(closest_title)+'\033[0m'+'?','\n')

    movie_index = get_index_from_title(closest_title)
    movie_list = list(enumerate(sim_matrix[int(movie_index)]))
    similar_movies = list(filter(lambda x:x[0] != int(movie_index), sorted(movie_list,key=lambda x:x[1], reverse=True)))


  for i,s in similar_movies[:how_many]:
    rec_movie.append(movies.iloc[i]['title'])

  return rec_movie

def multiple_movies_CB(list_of_movies):
  temp_reccomendation = []
  for movie in list_of_movies:
    temp_reccomendation.append(contents_based_recommender(movie, 20))
  flat_list = [item for sublist in temp_reccomendation for item in sublist]
  list_of_recommendation = []
  repeated_movies = []
  for i in flat_list:
    if i not in list_of_recommendation:
      list_of_recommendation.append(i)
  return list_of_recommendation
  
def find_title_db(list_of_names):
  ret = []
  for movie in list_of_names:
    if ',' in movie:
      movie = movie.split(',')
      ret.append(movieDB.loc[movieDB['original_title'].str.contains(movie[0])])
    else:
      ret.append(movieDB.loc[movieDB['original_title'].str.contains(movie)])

  list_of_row = [ele for ele in ret if len(ele['original_title']) != 0]
  return list_of_row

def get_movie_info(movieId):
  api_key = '84bd3118796019969d2fee13a58bcf90'
  url = f"https://api.themoviedb.org/3/movie/{movieId}?api_key={api_key}&language=pt-BR"
  image_url=f"https://image.tmdb.org/t/p/original"

  headers = {
      "accept": "application/json",
      # "Authorization": f"Bearer {reading_token}"
  }
  info = {}
  response = requests.get(url)
  if response.status_code ==200:
    response = json.loads(response.text)
    info['poster_path'] = image_url+response['poster_path']
    info['overview'] = response['overview']

    return info
  if response.status_code==404:
    return "movie not found"

def format_json(list_of_movies):
  response = []
  for movie in list_of_movies:
    id = int(movie['id'].values[0])
    info = get_movie_info(id)
    if info !='movie not found':
      dic = {'id': id,
              'original_title': movie['original_title'].values[0],
              'overview':info['overview'],
              'genres': movie['genres'].values[0],
              'poster_path': info['poster_path']}
      response.append(dic)
  return response
  
def lambda_handler(event, _):
    movies = event['body']
    rec = multiple_movies_CB(movies)
    titles = find_title_db(rec)
    response = format_json(titles)

    return response