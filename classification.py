import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load data using sklearn dataset
data = load_breast_cancer()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

# implement RandomForest to train our test set
model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)




