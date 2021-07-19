from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np

def get_data(train_samples = 5000):
    X,y=fetch_openml('mnist_784',version=1, return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=10000)

def train_model():
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        C=50. / train_samples, penalty='l1', solver='saga', tol=0.1
    )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    # print('Best C % .4f' % clf.C_)
    print("Test score with L1 penalty: %.4f" % score)

def predict_model():
    return clf.predict()

if __name__ == __main__: