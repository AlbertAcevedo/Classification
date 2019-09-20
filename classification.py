import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
# load data using sklearn dataset
data = pd.read_csv("APFNBTraining.csv")
y = data['90'] # target
X = data.drop(['90', 'Producto', 'Contrato'], axis = 1) #

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print("\ntrain:\n")
print(X_train.shape)
print(y_train.shape)

print("\ntest:\n")
print(X_test.shape)
print(y_test.shape)

# implement RandomForest to train our test set
model = GaussianNB()
model.fit(X_train, y_train)

print("score:", model.score(X_test, y_test))

# load data to predict

pred = pd.read_csv("APFNBPredict.csv")

predfnb = pred.drop(['Producto', 'Contrato'], axis=1)
print("\nFNB:\n")
print(predfnb.shape)

predictions = model.predict(predfnb)


predfnb['target'] = predictions
predfnb.to_csv('predictions.csv')

