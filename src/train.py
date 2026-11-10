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



#sigmoid func starts here
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

#lables takes an array of lables
#preds take an array of predicitiosn
def log_loss(labels, preds):
    #handle log instances hwere its undefined  
    preds = np.clip(preds, 1e-15, 1 - 1e-15)
    #formual: ll = -mean( Y x log(p) + (1 - y) x log(1-p))
    log_loss = labels * np.log(preds) + (1 - labels) * np.log(1-preds)
    return -np.mean(log_loss)


#graident descent 
def graidents(features, labels, weights, bias):
    z = features @ weights + bias 
    p = sigmoid(z)

    error = p - labels
    dw = (features.T @ error) / len(labels)
    db = np.mean(error)

    return dw, db

def train_model(feature, label, learning_rate, num_iters):
    weight = np.zeros(feature.shape[1]) #pulls num of fetaures and makes them all zeros
    bias = 0.0

    for i in range(num_iters):
        dw, db = graidents(feature, label, weight, bias)
        weight = weight - learning_rate * dw
        bias = bias - learning_rate * db

        if i % 100 == 0:
            z = feature @ weight + bias
            p = sigmoid(z)
            print(i, log_loss(label,p))


    return weight, bias


#returns the weight and bias of the model on 1000 runs
w,b = train_model(X_train, Y_train, 0.2, 1000)
print(w,b)