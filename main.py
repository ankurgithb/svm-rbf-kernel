import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles
X, y = make_circles(100, factor=.1, noise=0.1)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='bwr')
plt.savefig("circles_dataset.png", dpi=300, bbox_inches="tight")

zero_one_colourmap = ListedColormap(('blue', 'red'))
def plot_decision_boundary(X, y, clf):
    plt.figure(figsize=(6, 6))
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                                 stop = X_set[:, 0].max() + 1, 
                                 step = 0.01),
                       np.arange(start = X_set[:, 1].min() - 1, 
                                 stop = X_set[:, 1].max() + 1, 
                                 step = 0.01))
  
    plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), 
                                             X2.ravel()]).T).reshape(X1.shape),
               alpha = 0.75, 
               cmap = zero_one_colourmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = (zero_one_colourmap)(i), label = j)
    plt.title('SVM Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.savefig("svm_decision_boundary.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_3d_plot(X, y):
    fig = plt.figure(figsize=(8, 6))

    r = np.exp(-(X ** 2).sum(1))  # Transformation via Radial Basis Function
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=100, cmap='bwr')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('r')

    return fig

fig = plot_3d_plot(X, y)
fig.savefig("circles_3d.png", dpi=300, bbox_inches="tight")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

rbf_classifier = SVC(kernel="rbf")
rbf_classifier.fit(X_train, y_train)
y_pred = rbf_classifier.predict(X_test)

print("accuracy score=", accuracy_score(y_test, y_pred))

plot_decision_boundary(X, y, rbf_classifier)

