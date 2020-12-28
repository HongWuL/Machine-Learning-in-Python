import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from perceptron.perceptron import Perceptron
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris['data'][:100]
y = iris['target'][:100]


y[y == 0] = -1
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

clf = Perceptron()
clf.fit(X_train,y_train)
y_p, w, b = clf.predict(X_test)
acc = accuracy_score(y_test,y_p)
print(acc)





