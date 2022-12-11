#import libraries
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd


from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn import utils


matches = pd.read_csv("EPL2022_16.csv") # index is already in the data
#new_games=pd.read_csv("new_games.csv")

#new_matches=new_games

matches=matches.replace({'FTR': {'A' : 0.0 ,'D' :1.0, 'H': 2.0}})



#new_matches=new_matches.replace({'FTR': {'A' : 0.0 ,'D' :1.0, 'H': 2.0}})



#Creating predictors

data = matches


#all column names

cols5 = ['FTR','B365H','B365D','B365A']


data = data[cols5].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


#data = data[cols5]


data = data.dropna(axis=0)

matches = data


#convert y values to categorical values
lab = preprocessing.LabelEncoder()


X_train,X_test,y_train,y_test = train_test_split(matches,matches["FTR"],test_size=0.1,shuffle=True)


y_train = lab.fit_transform(y_train)
y_test = lab.fit_transform(y_test)

X_train = X_train.drop("FTR",axis=1)
X_test = X_test.drop("FTR",axis=1)




#define and fit the model
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

#make predictions

y_pred = clf.predict(X_test)


#evaluate the model
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)

y_pred = y_pred.flatten()

combined = pd.DataFrame(dict(actual=y_test, SVM=y_pred))

print(pd.crosstab(index=combined["actual"], columns=combined["SVM"]))


"""
merged = pd.concat([combined, X_test], axis=1)

merged = merged.drop('actual', axis=1)

print(merged)

"""