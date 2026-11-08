import pandas as pd
import numpy as np

features = ["score_differential", "game_seconds_remaining", "down", "ydstogo",
            "yardline_100", "posteam_timeouts_remaining", "defteam_timeouts_remaining"]


df = pd.read_parquet(f"data/plays_clean.parquet")

train = df[df["season"] <= 2022]
test = df[df["season"] >= 2023]

print(len(train), len(test))

#standardizing features to put them all on the same scale 
mean = train[features].mean()
std = train[features].std()

X_train = (train[features] - mean) / std
X_test = (test[features] - mean) / std

Y_train = train["posteam_won"]
Y_test = test["posteam_won"]

print(X_train.mean())
print(X_train.std())

#convert to arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

print(X_train.shape, Y_train.shape)