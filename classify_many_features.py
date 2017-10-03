import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVR, SVC, NuSVC, NuSVR
from sklearn.multiclass import OneVsOneClassifier
import os
import matplotlib.pyplot as plt
import math
from numpy import inf

class SVMFit_Exception(Exception):
    "This class gives customized exceptions for SVMFit class"

class Classify(object):

    def __init__(self):
        # create class variables
        self.header = ['f1', 'f2', 'f1c', 'f2c','hnr','pth','f2f1', 'f3f1']
        self.path = '/home/ribex/Desktop/Dev/voicetherapy_research/features2'
        self.filenames = [z for z in os.listdir(self.path) if z.endswith('.csv')]
        self.fullfilenames = [os.path.join(self.path, z) for z in os.listdir(self.path) if z.endswith('.csv')]
        self.X = None
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

class SVMFit(Classify):

    def __init__(self):
        super(SVMFit, self).__init__()

    def get_features(self, filename, features, data, length=None, reshuffle=True):
        """uses the features-list in features to extract selected data in the data dict
        """

        data_length = length
        for feature in features:
            if feature not in self.header:
                raise SVMFit_Exception('one or more feature not found in features in data set')
        feature_set = None
        if data_length is None:
            # go with the length of the features
            # recall that each dictionary carries a pandas dataframe
            data_length = data[filename].shape[0]
        feature_set = np.zeros([data_length, len(features)])
        for i in range(0, len(features)):
            feature_set[:, i] = np.asarray(data[filename][features[i]][0:data_length])
        if not reshuffle:
            return feature_set
        np.random.shuffle(feature_set)
        return feature_set

    def form_x(self, extracted_features, features, tr):
        """ *_tr_* refers to the training set. *_ts_* refers to the test set
        """
        # Define variables to hold the feature sets
        tr_lens = [int(np.floor(t.shape[0] * tr)) for t in extracted_features]

        tr_total_len = sum(tr_lens)
        X_tr = np.zeros([tr_total_len, len(features)])
        ts_lens = [int(np.floor(t.shape[0] * (1 - tr))) for t in extracted_features]

        ts_total_len = sum(ts_lens)
        X_ts = np.zeros([ts_total_len, len(features)])
        y_tr = np.zeros([tr_total_len, 1])
        y_ts = np.zeros([ts_total_len, 1])

        idx_tr = 0
        idx_ts = 0
        i = 0
        while (i < len(extracted_features)) :
            # Update the training feature set
            X_tr[idx_tr:idx_tr + tr_lens[i], :]  = extracted_features[i][0:tr_lens[i], :]
            y_tr[idx_tr:idx_tr + tr_lens[i], 0] = i
            # set the training position to the last index
            idx_tr = idx_tr + tr_lens[i]

            # Now update the test feature set
            X_ts[idx_ts:idx_ts + ts_lens[i], :] = extracted_features[i][tr_lens[i]:tr_lens[i] + ts_lens[i], :]
            y_ts[idx_ts:idx_ts + ts_lens[i], 0] = i
            # set the test index position to the last value to be extracted
            idx_ts = idx_ts + ts_lens[i]
            i = i+1
        return X_tr, y_tr.reshape(len(y_tr,)), X_ts, y_ts.reshape(len(y_ts),)

    def svmfit(self, features):

        tr = 0.7
        data = self.all_features()
        shortfiles = ['v.wav.csv','Yeye.wav.csv','myohmy.wav.csv','mymymy.wav.csv','mememe.wav.csv']
        longfiles = []

        num_of_samples = len(shortfiles)
        np.random.seed(4)

        # save statistics
        stats = {}
        length = None

        v = self.get_features(shortfiles[0], features, data, length)
        yeye = self.get_features(shortfiles[1], features, data, length)
        myohmy = self.get_features(shortfiles[2], features, data, length)
        mymymy = self.get_features(shortfiles[3], features, data, length)
        mememe = self.get_features(shortfiles[4], features, data, length)

        extracted_features = [v, yeye, myohmy, mymymy, mememe]
        X_tr, y_tr, X_ts, y_ts = self.form_x(extracted_features, features, tr)

        ###########################################################################################

        clf = OneVsOneClassifier(SVC(C=1, cache_size=400, coef0=0.0, degree=5, gamma='auto',
                                      kernel='rbf', max_iter=-1, shrinking=True, tol=.01, verbose=False), -1).fit(X_tr, y_tr)
        pred = clf.predict(X_ts)
        dec_func = pd.DataFrame(OneVsOneClassifier.decision_function(clf, X_ts))
        rmse, corr = self.calculate_stats(y_ts, pred)
        accuracy = float(sum(y_ts == pred)) / len(y_ts)

        print str(features), ':'
        print "accuracy: ", accuracy
        print "rmse: ", rmse
        print "corr: ", corr[0,1]
        dec_func.plot()
        plt.show()

    def permutate_features(self, all_features):
        pass

    def get_filelength(filename, data):
        return data[filename].shape[0]

    def shuffle(self, data):
        num_rows = 6

    def calculate_stats(self, act_x, pred_x):
        rmse = math.sqrt(((act_x - pred_x) ** 2).sum() / act_x.shape[0])
        corr = np.corrcoef(act_x, pred_x)
        return rmse, corr

def main():
    svmft = SVMFit()
    f = svmft.svmfit(['pth'])
    #print f

if '__main__' == __name__:
    main()