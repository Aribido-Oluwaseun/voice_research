print(__doc__)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

class Classify(object):

    def __init__(self):
        # create class variables
        self.header = ['f1', 'f2', 'f1c', 'f2c','hnr','pth','f2f1', 'f3f1']
        self.path = '/home/ribex/Desktop/Dev/voicetherapy_research/features2'
        self.filenames = [z for z in os.listdir(self.path) if z.endswith('.csv')]
        self.fullfilenames = [os.path.join(self.path, z) for z in os.listdir(self.path) if z.endswith('.csv')]


    def read_file(self, filename):
        data = pd.read_csv(filepath_or_buffer=filename)
        data = pd.DataFrame(data.values, columns=self.header)
        return data

    def all_features(self):
        data = {}
        for i in range(len(self.filenames)):
            data[self.filenames[i]] = self.read_file(filename=self.fullfilenames[i])
        return data

    def partition_data(self,data):
        """This file data into training set and test set
        The data structure is a dictionary
        """
        pass




def make_meshgrid(x, y, h=1):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
cl = Classify()
data = cl.all_features()
shortfiles = ['v.wav.csv','Yeye.wav.csv','myohmy.wav.csv','mymymy.wav.csv','mememe.wav.csv']
trunc = 100
v = np.asarray(data['v.wav.csv']['f1'][0:trunc])
yeye = np.asarray(data['Yeye.wav.csv']['f1'][0:trunc])
v2 = np.asarray(data['v.wav.csv']['f1c'][0:trunc])
yeye2 = np.asarray(data['Yeye.wav.csv']['f1c'][0:trunc])

X = np.zeros([trunc*2,2])

X[0:trunc,0] = v
X[trunc:trunc*2,0] = yeye
X[0:trunc,1] = v2
X[trunc:trunc*2,1] = yeye2

y = np.zeros([trunc*2,])
y[0:trunc] = 0
y[trunc:trunc*2] = 1

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 0.8  # SVM regularization parameter
models = (svm.NuSVR(kernel='rbf', nu=0.5, tol=.5),
          svm.SVC(kernel='linear', C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('F1c vs F2c features classification RBF kernel - Yeye/v',
          'F1c vs F2c features classification with polynomial - Yeye/v (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Formants 1c')
    ax.set_ylabel('Formants 2c')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
