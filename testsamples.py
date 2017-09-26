print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import numpy as np
import pandas as pd
from sklearn import svm
import os
import matplotlib.pyplot as plt

class Classify(object):

    def __init__(self):
        # create class variables
        self.header = ['f1', 'f2', 'f1c', 'f2c','hnr','pth']
        self.path = '/home/ribex/Desktop/Dev/voicetherapy_research/features'
        self.filenames = [z for z in os.listdir(self.path) if z.endswith('.csv')]
        self.fullfilenames = [os.path.join(self.path, z) for z in os.listdir(self.path) if z.endswith('.csv')]
        pass

    def read_file(self, filename):
        data = pd.read_csv(filepath_or_buffer=filename)
        data = pd.DataFrame(data.values, columns=self.header)
        return data

    def all_features(self):
        data = {}
        for i in range(len(self.filenames)):
            data[self.filenames[i]] = self.read_file(filename=self.fullfilenames[i])
        return data

    def svmfit(self):
        data = self.all_features()

        shortfiles = ['v.wav.csv','Yeye.wav.csv','myohmy.wav.csv','mymymy.wav.csv','mememe.wav.csv']
        trunc = 20
        v = np.asarray(data['v.wav.csv']['pth'][0:trunc])
        yeye = np.asarray(data['Yeye.wav.csv']['pth'][0:trunc])
        myohmy = np.asarray(data['myohmy.wav.csv']['pth'][0:trunc])
        mymymy = np.asarray(data['mymymy.wav.csv']['pth'][0:trunc])
        mememe = np.asarray(data['mememe.wav.csv']['pth'][0:trunc])

        X = np.zeros((2,trunc))

        X[0,:] = v
        X[1,:] = yeye
        Y = np.asarray([0,1])


        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(X, Y)
        print clf.fit(X, Y)

def make_meshgrid(x, y, h=2):
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
trunc = 20
v = np.asarray(data['v.wav.csv']['pth'][0:trunc])
yeye = np.asarray(data['Yeye.wav.csv']['pth'][0:trunc])
v2 = np.asarray(data['v.wav.csv']['f1'][0:trunc])
yeye2 = np.asarray(data['Yeye.wav.csv']['f1'][0:trunc])

X = np.zeros([trunc*2,2])

X[0:trunc,0] = v
X[trunc:trunc*2,0] = yeye
X[0:trunc,1] = v2
X[trunc:trunc*2,1] = yeye2

y = np.zeros([trunc*2,])
y[0:trunc] = 0
y[trunc:trunc*2] = 1

#iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
#X = iris.data[:, :2]


#y = iris.target


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = svm.SVC(kernel='rbf', gamma=0.7, C=C)
models = models.fit(X, y)

# title for the plots
titles = ('SVC with RBF kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(sub.flatten(), models, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
sub.flatten().scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
sub.flatten().set_xlim(xx.min(), xx.max())
sub.flatten().set_ylim(yy.min(), yy.max())
sub.flatten().set_xlabel('Sepal length')
sub.flatten().set_ylabel('Sepal width')
sub.flatten().set_xticks(())
sub.flatten().set_yticks(())
sub.flatten().set_title('SVC with RBF kernel')

plt.show()
