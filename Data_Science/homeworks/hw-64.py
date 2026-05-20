import pandas as pd
from read_table import read_table
from merge import merge

unames = ["user_id", "gender", "age", "occupation", "zip"]
users = pd.read_table("examples/users.dat", sep="::",
header=None, names=unames, engine="python")

rnames = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_table("examples/ratings.dat", sep="::",
header=None, names=rnames, engine="python")

mnames = ["movie_id", "title", "genres"]
movies = pd.read_table("examples/movies.dat", sep="::",
header=None, names=mnames, engine="python")

# print(users.head())
# print(ratings.head())
# print(movies.head())

data = pd.merge(pd.merge(ratings, users), movies)
# print(data.head())

# print(data.iloc[0])

mean_ratings = data.pivot_table("rating", index="title",
columns="gender", aggfunc="mean")
# print(mean_ratings.head(5))

ratings_by_title = data.groupby("title").size()
# print(ratings_by_title.head())

active_titles = ratings_by_title.index[ratings_by_title >= 250]
# # print(active_titles)

mean_ratings = mean_ratings.loc[active_titles]
# print(mean_ratings.head())

top_female_ratings = mean_ratings.sort_values("F", ascending=False)
# print(top_female_ratings.head())

# mean_ratings["diff"] = mean_ratings["M"] - mean_ratings["F"]
# sorted_by_diff = mean_ratings.sort_values("diff")
# print(sorted_by_diff.head())
# print(sorted_by_diff[::-1].head())

# rating_std_by_title = data.groupby("title")["rating"].std()
# rating_std_by_title = rating_std_by_title.loc[active_titles]
# # print(rating_std_by_title.head())
# print(rating_std_by_title.sort_values(ascending=False)[:10])

# print(movies["genres"].head())
# # print(movies["genres"].head().str.split("|"))

movies["genre"] = movies.pop("genres").str.split("|")
# print(movies.head())

movies_exploded = movies.explode("genre")
# print(movies_exploded[:10])

ratings_with_genre = pd.merge(pd.merge(movies_exploded, ratings), users)
print(ratings_with_genre.iloc[0])

genre_ratings = (ratings_with_genre.groupby(["genre", "age"])
                    ["rating"].mean()
                    .unstack("age"))

print(genre_ratings[:10])

print('_____________________ MY _____________________')
unames = ["user_id", "gender", "age", "occupation", "zip"]
users = read_table("examples/users.dat", sep="::",
header=None, names=unames)

rnames = ["user_id", "movie_id", "rating", "timestamp"]
ratings = read_table("examples/ratings.dat", sep="::",
header=None, names=rnames)

mnames = ["movie_id", "title", "genres"]
movies = read_table("examples/movies.dat", sep="::",
header=None, names=mnames)

# print(users.head())
# print(ratings.head())
# print(movies.head())

data = merge(merge(ratings, users), movies)
# print(data.head())

# print(data.iloc[0])

mean_ratings = data.pivot_table("rating", index="title",
columns="gender", aggfunc="mean")
# print(mean_ratings.head(5))

ratings_by_title = data.groupby("title").size()
# print(ratings_by_title.head())

active_titles = ratings_by_title.index[ratings_by_title >= 250]
# # print(active_titles)

mean_ratings = mean_ratings.loc[active_titles]
# print(mean_ratings.head())

top_female_ratings = mean_ratings.sort_values("F", ascending=False)
# print(top_female_ratings.head())

# mean_ratings["diff"] = mean_ratings["M"] - mean_ratings["F"]
# sorted_by_diff = mean_ratings.sort_values("diff")
# print(sorted_by_diff.head())
# print(sorted_by_diff[::-1].head())

# rating_std_by_title = data.groupby("title")["rating"].std()
# rating_std_by_title = rating_std_by_title.loc[active_titles]
# print(rating_std_by_title.head())
# print(rating_std_by_title.loc['101 Dalmatians (1961)'])
# print(rating_std_by_title.sort_values(ascending=False)[:10])

# print(movies["genres"].head())
# # print(movies["genres"].head().str.split("|"))

movies["genre"] = movies.pop("genres").str.split("|")
# print(movies.head())

movies_exploded = movies.explode("genre")
# print(movies_exploded[:10])

ratings_with_genre = merge(merge(movies_exploded, ratings), users)
# print(ratings_with_genre.iloc[0])

genre_ratings = (ratings_with_genre.groupby(["genre", "age"])
                    ["rating"].mean()
                    .unstack("age"))

print(genre_ratings[:10])