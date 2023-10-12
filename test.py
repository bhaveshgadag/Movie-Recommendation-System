import pandas as pd

movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')
ratings = pd.read_csv('ratings.csv')

print(movies.head())
print(credits.head())
print(ratings.head())
