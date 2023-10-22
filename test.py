import pandas as pd

movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')
ratings = pd.read_csv('ratings.csv')

# Calculate Weighted Rating
# WR = (v / (v + m)) * R + (m / (v + m)) * C
# v - number of votes for a movie
# m - minimum number of votes required
# R - average rating of the movie
# C - average rating across all movies

m = movies['vote_count'].quantile(0.9)
C = movies['vote_average'].mean()

def weighted_average(df, m=m, C=C):
    R = df['vote_average']             
    v = df['vote_count']
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    return WR

movies['weigthed_ratings'] = movies.apply(weighted_average, axis=1)
print(movies.head())

