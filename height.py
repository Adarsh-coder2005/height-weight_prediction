import numpy as np
import sklearn
import pandas as pd
from sklearn import linear_model
import pickle

df = pd.read_csv('data.csv')
print(df.head(10))

x = np.array(df.drop(['Weight'], 1))
y = np.array(df['Weight'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    lr = linear_model.LinearRegression()
    lr.fit(x_train, y_train)

    acc = lr.score(x_test, y_test)
    print(acc)

    if acc>best:
        best = acc
        with open('height.pickle', 'wb') as f:
            pickle.dump(lr,f)'''

file = open('height.pickle', 'rb')
lr = pickle.load(file)

predict = lr.predict(x_test)

for i in range(len(predict)):
    print(predict[i], x_test[i], y_test[i])