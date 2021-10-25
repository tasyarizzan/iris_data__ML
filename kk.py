
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier

dataset=load_iris()
data=pd.DataFrame(dataset['data'],columns=["Petal length","Petal Width","Sepal Length","Sepal Width"])
data['Species']=dataset['target']
data['Species']=data['Species'].apply(lambda x: dataset['target_names'][x])
print (data.head())
#splittnig 70x30
train,test=train_test_split(data,test_size=0.3)
print(train.shape, test.shape)

train_X=train[['Sepal Length',"Sepal Width","Petal length","Petal Width"]]
train_y=train.Species
test_X=test[['Sepal Length',"Sepal Width","Petal length","Petal Width"]]
test_y=test.Species


dtmodel=DecisionTreeClassifier()
dtmodel.fit(train_X,train_y)
dtpredict=dtmodel.predict(test_X)
dtaccuracy=metrics.accuracy_score(dtpredict,test_y)
print("Decision Tree Model Accuracy is {}".format(dtaccuracy*100))

test_preddf=test.copy()
test_preddf['Predicted Species']=dtpredict
wrongpred=test_preddf.loc[test['Species'] != dtpredict]
print(wrongpred)

#Support VC
lin_svc = svm.SVC (kernel = 'linear'). fit (train_X, train_y) # kernel kernel function is a linear function of the nuclear
rbf_svc = svm.SVC (kernel = 'rbf'). fit (train_X, train_y) # kernel is the kernel function
poly_svc = svm.SVC (kernel = 'poly', degree = 3) .fit (train_X, train_y) #kernel a polynomial kernel function

lin_svc_pre = lin_svc.predict (test_X) # linear prediction kernel function svm
rbf_svc_pre = rbf_svc.predict (test_X) # forecast radial basis kernel function rbf
poly_svc_pre = poly_svc.predict (test_X) # prediction polynomial kernel function

# Score function, return the correct rate based on the given data and the tab
acc_lin_svc = lin_svc.score(test_X,test_y)
acc_rbf_svc = rbf_svc.score(test_X,test_y)
acc_poly_svc = poly_svc.score(test_X,test_y)

print('acc_lin_svc: ',acc_lin_svc)
print('acc_lin_predicted: ',lin_svc_pre)
print('acc_rbf_svc: ',acc_rbf_svc)
print('acc_rbf_predicted: ',rbf_svc_pre)
print('acc_poly_svc: ',acc_poly_svc)
print('acc_poly_predicted: ',poly_svc_pre)

# Fitting Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(train_X, train_y)
y_pred = nvclassifier.predict(test_X)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, y_pred)
print(cm)

a = cm.shape
corrPred = 0
falsePred = 0

for row in range(a[0]):
    for c in range(a[1]):
        if row == c:
            corrPred +=cm[row,c]
        else:
            falsePred += cm[row,c]
print('Correct predictions: ', corrPred)
print('False predictions', falsePred)
print ('\n\nAccuracy of the Naive Bayes Clasification is: ', corrPred/(cm.sum()))