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

print(similarity_matrix)
