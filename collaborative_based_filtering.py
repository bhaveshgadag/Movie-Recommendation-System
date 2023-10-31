import pandas as pd
from surprise import Dataset, Reader, SVD, model_selection

ratings = pd.read_csv("ratings.csv")

# print(ratings.head())

reader = Reader(rating_scale=(1,5))

dataset = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# print(dataset)

trainset = dataset.build_full_trainset()

# print(list(trainset.all_ratings()))

svd = SVD()
svd.fit(trainset)

print(svd.predict(15, 1956))
print(svd.predict(15, 1956).est)

# Validation

print(model_selection.cross_validate(svd, dataset, measures=['RMSE', 'MAE']))
