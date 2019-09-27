from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
import pandas as pd

# load data using sklearn dataset
df = pd.read_csv("APFNBTraining.csv")

# split X into majority and minority
X_majority = df[df['90'] == 0]
X_minority = df[df['90'] == 1]



# upsample minority class
X_min_upsample = resample(X_minority,
                          replace=True,
                          n_samples= 30000,
                          random_state=42)

# split your data
data = pd.concat([X_majority, X_min_upsample])
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

# implement Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

print("score:", model.score(X_test, y_test))

# load data to predict, separate label columns to later concatenate with predictions
pred = pd.read_csv("APFNBPredict.csv")
predfnb = pred.drop(['Producto', 'Contrato'], axis=1)
print("\nFNB:\n")
print(predfnb.shape)

# predict and then send predictions to a csv file
predictions = model.predict(predfnb)
predfnb['target'] = predictions
predfnb.to_csv('predictions.csv')
