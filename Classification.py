# Recursive Feature Elimination
import numpy as np
import pandas as pd
from InformtaionGain import ig_list
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score


# load cancer datasets
df = pd.read_csv("data.csv", header = 0)
original_headers = list(df.columns.values)

# extract numeric columns
numericColumns = df._get_numeric_data()



# load the iris datasets
dataset = datasets.load_iris()

#21
target=numericColumns["classifierColumn"]


df = pd.read_csv("out2.csv", header = 0)
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x) #scale x data between o and 1
numericColumns = pd.DataFrame(x_scaled) # dataframs with x values scaled and columns with numeric index





#Apply SVM

X_train, X_test, y_train, y_test = train_test_split(numericColumns, target, test_size=0.1, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)


#print(clf.predict(X_test[20:100]))

for k in range (1):
    X_train, X_test, y_train, y_test = train_test_split(numericColumns, target, test_size=0.3, random_state=0)
    clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(clf.predict(X_test[1:4]))
    print(y_test[1:4])
    print(scores)

