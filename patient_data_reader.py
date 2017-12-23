#see comments in file topicrnn.py to code this patientreader file
import os
import itertools
import numpy as np
import nltk
import math
from util import *

max_visit_size = 300


class PatientReader(object):
    def __init__(self, config):
        self.data_path = data_path = config.data_path

        self.vocab_path = vocab_path = os.path.join(data_path, "vocab.pkl")

        # use train data to build vocabulary
        if os.path.exists(vocab_path):
            self._load()
        else:
            pass

        self.vocab_size = config.vocab_size
        self.n_train_patients = math.ceil((len(self.X_train_data) + 0.0))
        self.n_valid_patients = math.ceil((len(self.X_valid_data) + 0.0))
        self.n_test_patients = math.ceil((len(self.X_test_data) + 0.0))

        self.lda_vocab_size = config.lda_vocab_size
        self.n_stops = config.n_stops

        self.idx2word = {v: k for k, v in self.vocab.items()} #needed to go from index to concept 

        print("vocabulary size: {}".format(self.vocab_size))
        print("number of training documents: {}".format(self.n_train_patients))
        print("number of validation documents: {}".format(self.n_valid_patients))
        print("number of testing documents: {}".format(self.n_test_patients))

    def _load(self):
        self.vocab = load_pkl(self.vocab_path)

        self.X_train_data = load_pkl(self.data_path + '/' + 'X_train' + '.pkl')
        self.Y_train_data = load_pkl(self.data_path + '/' + 'Y_train' + '.pkl')

        self.X_valid_data = load_pkl(self.data_path + '/' + 'X_valid' + '.pkl')
        self.Y_valid_data = load_pkl(self.data_path + '/' + 'Y_valid' + '.pkl')

        self.X_test_data = load_pkl(self.data_path + '/' + 'X_test' + '.pkl')
        self.Y_test_data = load_pkl(self.data_path + '/' + 'Y_test' + '.pkl')

    def get_data_from_type(self, data_type):
        if data_type == "train":
            X_raw_data = self.X_train_data
            Y_raw_data = self.Y_train_data
        elif data_type == "valid":
            X_raw_data = self.X_valid_data
            Y_raw_data = self.Y_valid_data
        elif data_type == "test":
            X_raw_data = self.X_test_data
            Y_raw_data = self.Y_test_data
        else:
            raise Exception(" [!] Unkown data type %s: %s" % data_type)

        return X_raw_data, Y_raw_data

    def get_Xc(self, data):
        """data is a patient...a sequence of visits
            so a list of lists...the outer list is of size T_patient
            the inner lists contain the concepts within each visit
        """
        patient = [concept for visit in data for concept in visit]
        patient = [x-1 for x in patient] 
        counts = np.bincount(patient, minlength=self.vocab_size)
        stops_flag = np.array(list(np.ones([self.lda_vocab_size], dtype=np.int32)) +
                              list(np.zeros([self.n_stops], dtype=np.int32)))

        return counts * stops_flag

    def get_X(self, data):
        """
        data is a list of lists of different length
        return an array of shape CxT where 
        entry Mij = ci if ci in visit j
        """
        T_patient = len(data)
        res = np.zeros([self.vocab_size, T_patient])
        for i in range(self.vocab_size):
            for j in range(T_patient):
                if (i+1) in data[j]:
                    res[i, j] = (i+1)

        return res

    def iterator(self, data_type="train"):
        """
        goes over the data and
        returns X, Xc, Y, and seq_len in a round robin
        seq_len is a vector of size C where each 
        entry is T_patient
        """
        X_raw_data, Y_raw_data = self.get_data_from_type(data_type)

        x_infos = itertools.cycle(([self.get_X(X_doc[:max_visit_size]), self.get_Xc(X_doc[:max_visit_size])]
                                   for X_doc in X_raw_data if X_doc != []))
        y_infos = itertools.cycle(([Y_doc[:max_visit_size], np.array([len(Y_doc[:max_visit_size])]*self.vocab_size)]
                                   for Y_doc in Y_raw_data if Y_doc != []))

        return x_infos, y_infos

