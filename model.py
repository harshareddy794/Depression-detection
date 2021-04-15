#  Import all the libraries for predictive modelling

import numpy as np #create arrays
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#  Data Cleaning
url='dataset.csv'
df = pd.read_csv(url)
train = df.drop(['Name','score','Risk','Age'], axis=1)
train= np.asarray(train, dtype='int64')
test = df[['Risk']]
test= np.asarray(test, dtype='int64')
test.shape
X_train, X_test, y_train, y_test = train_test_split(train,test, test_size=0.35, random_state=2)
reg = LogisticRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
n =[[1,0,1,0,1,0]]
o = reg.predict(n)
print(reg.score(X_test, y_test))
X_test.shape
#Accuracy 88.65%

pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))