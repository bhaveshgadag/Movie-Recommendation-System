import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pd.read_csv("movies_small.csv", sep=';')
# print(movies)

# Term Frequency Inverse Document frequeny
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# print(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()))

# Similarity Matrix
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# print(similarity_matrix)

def similar_movies(movie_title, no_of_movies):
    index = movies.loc[movies['title']==movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movies_index = [tpl[0] for tpl in similarity_scores[1:no_of_movies + 1]]
    similar_movies = list(movies['title'].iloc[similar_movies_index])
    return similar_movies

print(similar_movies('Cars 2', 2))
