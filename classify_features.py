import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVR, SVC, NuSVC, NuSVR
from sklearn.multiclass import OneVsOneClassifier
import os
import matplotlib.pyplot as plt
import math
from numpy import inf

class Classify(object):

    def __init__(self):
        # create class variables
        self.header = ['f1', 'f2', 'f1c', 'f2c','hnr','pth','f2f1', 'f3f1']
        self.path = '/home/ribex/Desktop/Dev/voice_research/voicetherapy_research/features2'
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
        total_length = 200
        trunc = int(total_length*0.7)
        ver = total_length - trunc

        num_of_features = 2
        num_of_samples = len(shortfiles)
        np.random.seed(4)

        # save statistics
        stats = {}
        for i in range(len(self.header)):

            v = np.asarray([data['v.wav.csv'][self.header[i]][0:total_length]]).reshape((total_length, 1))
            np.nan_to_num(v)
            np.random.shuffle(v)
            yeye = np.asarray([data['Yeye.wav.csv'][self.header[i]][0:total_length]]).reshape((total_length, 1))
            np.nan_to_num(yeye)
            np.random.shuffle(yeye)
            myohmy = np.asarray([data['myohmy.wav.csv'][self.header[i]][0:total_length]]).reshape((total_length, 1))
            np.nan_to_num(myohmy)
            np.random.shuffle(myohmy)
            mymymy = np.asarray([data['mymymy.wav.csv'][self.header[i]][0:total_length]]).reshape((total_length, 1))
            np.nan_to_num(mymymy)
            np.random.shuffle(mymymy)
            mememe = np.asarray([data['mememe.wav.csv'][self.header[i]][0:total_length]]).reshape((total_length, 1))
            np.nan_to_num(mememe)
            np.random.shuffle(mememe)


            # Re-shuffle data
            v_tr = v[0:trunc,:]
            yeye_tr = yeye[0:trunc,:]
            myohmy_tr = myohmy[0:trunc,:]
            mymymy_tr = mymymy[0:trunc,:]
            mememe_tr = mememe[0:trunc,:]

            v_ts = v[trunc:total_length,:]
            yeye_ts = yeye[trunc:total_length]
            myohmy_ts = myohmy[trunc:total_length]
            mymymy_ts = mymymy[trunc:total_length]
            mememe_ts = mememe[trunc:total_length]

            X = np.zeros([trunc*num_of_samples, num_of_features])
            X_ts = np.zeros([ver*num_of_samples, num_of_features])

            y = np.zeros([len(X),1])
            y_ts = np.zeros([len(X_ts),1])

            selected_data = [v_tr, yeye_tr, myohmy_tr, mymymy_tr, mememe_tr]
            test_data = [v_ts, yeye_ts, myohmy_ts, mymymy_ts, mememe_ts]

            init_tr = 0
            init_ts = 0
            for j in range(len(selected_data)):
                X[init_tr : (j+1) * trunc, 0:2] = selected_data[j]
                X_ts[init_ts: (j + 1) * ver, 0:2] = test_data[j]
                y[init_tr : (j + 1) * trunc, 0] = j
                y_ts[init_ts: (j + 1) * ver, 0] = j
                init_tr = trunc*(j+1)
                init_ts = ver*(j+1)


            # change y, y_ts back to 1-dimension
            y = y.reshape((len(X),))
            y_ts = y_ts.reshape((len(X_ts,)))

            clf = OneVsOneClassifier(SVC(C=1, cache_size=400, coef0=0.0, degree=5, gamma='auto',
                                         kernel='rbf', max_iter=-1, shrinking=True, tol=.01, verbose=False),-1).fit(X, y)
            pred = clf.predict(X_ts)
            dec_func = pd.DataFrame(OneVsOneClassifier.decision_function(clf, X_ts))
            rmse, corr = self.calculate_stats(y_ts,pred)
            accuracy = float(sum(y_ts == pred))/len(y_ts)
            stats.update({self.header[i]:[accuracy, rmse, corr[0,1]]})

            # print "accuracy: ", accuracy
            # print "rmse: ", rmse
            # print "corr: ", corr[0,1]
            #dec_func.plot()
            #plt.show()
        print stats

    def shuffle(self, data):
        num_rows = 6

    def calculate_stats(self, act_x, pred_x):
        rmse = math.sqrt(((act_x - pred_x) ** 2).sum() / act_x.shape[0])
        corr = np.corrcoef(act_x, pred_x)
        return rmse, corr

def main():
    cl = Classify()
    f = cl.svmfit()
    #print f

if '__main__' == __name__:
    main()