##
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# plot function
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[0]),
           yticks=np.arange(cm.shape[1]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(i, j, format(cm[i, j], fmt),
                    horizontalalignment="center", verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.tight_layout()
    return ax


##
# load data using sklearn dataset
df = pd.read_csv("APFNBTraining.csv")


# split X into majority and minority
X_majority = df[df['90'] == 0]
X_minority = df[df['90'] == 1]

##
# prepare data set, minority class upsampling is enabled
X_min_upsample = resample(X_minority,
                          replace=True,
                          n_samples= 15000,
                          random_state=42)

# split your data
data = pd.concat([X_majority, X_min_upsample])
y = data['90'] # target
X = data.drop(['90', 'Producto', 'Contrato'], axis=1) #
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


print("\ntrain:\n")
print(X_train.shape)
print(y_train.shape)

print("\ntest:\n")
print(X_test.shape)
print(y_test.shape)

##
# plot features
dplt = df.drop(['90', 'Producto', 'Contrato'], axis=1)
for i in range(len(dplt.columns)):
    dplt.iloc[:, i].hist()
    tit = dplt.columns[i]
    axis = plt.subplot()
    axis.set(title=tit)
    plt.tight_layout()
    plt.show()

##
# implement Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

print("score:", model.score(X_test, y_test))

##
# confussion matrix
predicted = model.predict(X_test)
print(metrics.classification_report(y_test, predicted))
cmat = metrics.confusion_matrix(predicted, y_test)
plot_confusion_matrix(predicted,
                      y_test,
                      normalize=False,
                      title="CM")
plt.show()

##
# load data to predict, separate label columns to later concatenate with predictions
pred = pd.read_csv("APFNBPredict.csv")
predfnb = pred.drop(['Producto', 'Contrato'], axis=1)
print("\nFNB:\n")
print(predfnb.shape)

# predict and then send predictions to a csv file
predictions = model.predict(predfnb)
predfnb['target'] = predictions
predfnb.to_csv('predictions.csv')
print(sum(y_test))
