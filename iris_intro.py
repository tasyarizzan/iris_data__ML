print('\n--------Part A ------------------\n\n')
print('Import the libraries.......')
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn
from sklearn import *

print('\n--------- Part A 3,4,5 ------------------\n\n')
iris = datasets.load_iris()
print("---number of data :\n", iris.data)
print("names of variables :\n", iris.feature_names)
print("names of variables :\n", iris.target_names)

print('\n--------- Part B Data normalization------------------\n\n')
X=[[1, -1, 2] ,[2, 0, 0], [0, 1, -1]]
Xmean=numpy.mean(X)
Xvar=numpy.var(X)
print("The matrix is: \n", X)
print("The mean is: \n", Xmean)
print("The variance is: \n", Xvar)
ScX=sklearn.preprocessing.scale(X)
print("Scaled matrix: \n",ScX)
print("The ScaledX mean is: \n", numpy.mean(ScX))
print("The ScaledX variance is: \n", numpy.var(ScX))

print('\n--------- Part C MinMax Normalization------------------\n\n')
#MMScaledX=MinMaxScaler.fit(X)
scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(X)
print(scaler.data_max_)
MMScaledX=scaler.transform(X)
print("MinMax normalized matrix: \n",MMScaledX)
print("The ScaledX mean is: \n", numpy.mean(MMScaledX))
print("The ScaledX variance is: \n", numpy.var(MMScaledX))

print('\n--------- Part D Data visualization -----------------\n\n')
iris_frame = pandas.DataFrame(iris.data)
iris_frame.columns = iris.feature_names
iris_frame['target'] = iris.target
iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])
print(iris_frame)

def plot_iris(f1, f2):
n_samples = len(iris.target)
for t in set(iris.target):
x = [iris.data[i,f1] for i in range(n_samples) if iris.target[i]==t]
y = [iris.data[i,f2] for i in range(n_samples) if iris.target[i]==t]
plt.scatter(x, y, color=['red', 'green', 'blue'][t], label=iris.target_names[t])
plt.xlabel(iris.feature_names[f1])
plt.ylabel(iris.feature_names[f2])
plt.title('Iris Dataset')
plt.legend(iris.target_names, loc='lower right')
plt.show()
n_features = len(iris.feature_names)
pairs = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]
for (f1, f2) in pairs:
plot_iris(f1, f2)

print("The correlation between each pair is: \n", iris_frame[['sepal length (cm)','sepal width
(cm)','petal length (cm)','petal width (cm)']].corr())

print('\n--------- Part E------------------\n\n')
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
% str(pca.explained_variance_ratio_))
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('LDA of IRIS dataset')
plt.show()
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], iris_frame[['target']], test_size = 0.3)
model = KMeans(n_clusters=3)
model.fit(train_data)
model_predictions = model.predict(test_data)
print (metrics.accuracy_score(test_labels, model_predictions)
print (metrics.classification_report(test_labels, model_predictions)

print('\n--------- Part F------------------\n\n')
from sklearn.cluster import AgglomerativeClustering
groups = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
groups .fit_predict(iris)
plt.scatter(iris['petal_length'], iris['petal_width'], c=groups.labels_, cmap='cool')