import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
h = .02  # step size in the mesh

np.random.seed(50) # class=6, seed=20
    
names = ["LogisticRegression","Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
           "Neural Net", "AdaBoost","Naive Bayes"]#"Decision Tree",

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025,probability=True),
    SVC(gamma=2, C=1,probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    ]

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                            random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)

# datasets = [make_moons(noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable
#             ]
df = pd.read_excel('data1.xlsx')#这个会直接默认读取到这个Excel的第一个表单
# df.columns = ["y","x1","x2","x3","x4","x5","x6"]
x1 = df["x1"].values
x2 = df["x2"].values
x3 = df["x3"].values
x4 = df["x4"].values
x5 = df["x5"].values
x6 = df["x6"].values
y = df["y"].values
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
X = np.array([x1,x2,x3,x4])#
X = X.transpose(1,0)
# X = min_max_scaler.fit_transform(X)
Y=y

	# creating testing and training set
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)


# iterate over datasets

    # preprocess dataset, split into training and test part
# X, y = ds
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))

# # just plot the dataset first
# cm = plt.cm.RdBu
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
# ax = plt.subplot(1, len(classifiers) + 1, 1)

# ax.set_title("Input data")
# # Plot the training points
# ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
#             edgecolors='k')
# # Plot the testing points
# ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
#             edgecolors='k')
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_xticks(())
# ax.set_yticks(())
# plt.tight_layout()
# plt.show()

    # iterate over classifiers
i=1
figure = plt.figure(figsize=(24, 10))
for name, clf in zip(names, classifiers):
    a=int(len(classifiers)/2 + 1)
    ax = plt.subplot(2, int(len(classifiers)/2)+1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name+'score is %0.2f'%score)
    # Plot the training points


    area_under_curve = roc_auc_score(y_test, clf.predict(X_test))#classifier.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])#classifier.predict_proba(X_test)[:,1]
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    ax.plot(fpr, tpr, label=name+' (area = %0.2f)' % area_under_curve)
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
	#plt.savefig('Log_ROC')


    i += 1

plt.tight_layout()
plt.show()
