import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

movies = pd.read_csv("movies_small.csv", sep=';')
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies['overview'])

print(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()))