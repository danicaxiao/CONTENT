#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function

import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne
from config import Config
from patient_data_reader import PatientReader
import os
import time
import numpy as np
from lasagne.layers.timefusion import MaskingLayer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, precision_recall_curve
from lasagne.layers.theta import ThetaLayer

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 200
# Number of training sequences in each batch


# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
num_epochs = 30


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    if shuffle:
        indices = np.arange(len(inputs[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]

def cut_sel(paddedMat, lengths):
    newVec = []
    for i in range(len(paddedMat)):
        newVec.extend(paddedMat[i,:].flatten()[:lengths[i][0]])
    return newVec

def main(data_sets):
    # Optimization learning rate
    LEARNING_RATE = theano.shared(np.array(0.001, dtype=theano.config.floatX))
    eta_decay = np.array(0.5, dtype=theano.config.floatX)
    # Min/max sequence length
    MAX_LENGTH = 300
    X_raw_data, Y_raw_data = data_sets.get_data_from_type("train")
    trainingAdmiSeqs, trainingMask, trainingLabels, trainingLengths, ltr = prepare_data(X_raw_data, Y_raw_data, vocabsize= 619, maxlen = MAX_LENGTH)
    Num_Samples, MAX_LENGTH, N_VOCAB = trainingAdmiSeqs.shape

    X_valid_data, Y_valid_data = data_sets.get_data_from_type("valid")
    validAdmiSeqs, validMask, validLabels, validLengths, lval  = prepare_data(X_valid_data, Y_valid_data, vocabsize= 619, maxlen = MAX_LENGTH)

    X_test_data, Y_test_data = data_sets.get_data_from_type("test")
    test_admiSeqs, test_mask, test_labels, testLengths, ltes = prepare_data(X_test_data, Y_test_data, vocabsize= 619, maxlen = MAX_LENGTH)
    alllength = sum(trainingLengths) + sum(validLengths) + sum(testLengths)
    print(alllength)
    eventNum = sum(ltr)+sum(lval)+sum(ltes)
    print(eventNum)



    print("Building network ...")
    N_BATCH = 100
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, N_VOCAB))
    # The network also needs a way to provide a mask for each sequence.  We'll
    # use a separate input layer for that.  Since the mask only determines
    # which indices are part of the sequence for each batch entry, they are
    # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
    embedsize = 100
    n_topics = 50
    l_embed = lasagne.layers.DenseLayer(l_in, num_units=embedsize, num_leading_axes=2)
    #l_drop = lasagne.layers.dropout(l_embed)
    l_forward = lasagne.layers.GRULayer(
        l_embed, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        only_return_final=False)

    l_forward = MaskingLayer([l_forward, l_mask])

    l_1 = lasagne.layers.DenseLayer(l_in, num_units=N_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=2)
    l_2 = lasagne.layers.DenseLayer(l_1, num_units=N_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=2)
    mu = lasagne.layers.DenseLayer(l_2, num_units=n_topics, nonlinearity=None, num_leading_axes=1)# batchsize * n_topic
    log_sigma = lasagne.layers.DenseLayer(l_2, num_units=n_topics, nonlinearity=None, num_leading_axes=1)# batchsize * n_topic
    l_theta = ThetaLayer([mu,log_sigma],maxlen = MAX_LENGTH)#batchsize * maxlen

    l_dense0 = lasagne.layers.DenseLayer(
        l_forward, num_units=1, nonlinearity=None,num_leading_axes=2)
    l_dense1 = lasagne.layers.reshape(l_dense0, ([0], [1]))#batchsize * maxlen
    l_dense = lasagne.layers.ElemwiseMergeLayer([l_dense1, l_theta],T.add)
    l_out = lasagne.layers.NonlinearityLayer(l_dense,nonlinearity=lasagne.nonlinearities.sigmoid)



    target_values = T.matrix('target_output')
    target_values_flat = T.flatten(target_values)

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    # The network output will have shape (n_batch, maxlen); let's flatten to get a
    # 1-dimensional vector of predicted values
    predicted_values = network_output.flatten()
    # Our cost will be mean-squared error
    cost = lasagne.objectives.binary_crossentropy(predicted_values, target_values_flat)
    kl_term = l_theta.klterm
    cost = cost.sum()+kl_term


    test_output = lasagne.layers.get_output(l_out, deterministic=True)

    #cost = T.mean((predicted_values - target_values)**2)
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adam(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var],cost)
    prd = theano.function([l_in.input_var, l_mask.input_var], test_output)

    # We'll use this "validation set" to periodically check progress


    print("Training ...")
    try:
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches_listinputs([trainingAdmiSeqs, trainingLabels, trainingMask], N_BATCH,
                                                        shuffle=True):
                inputs = batch
                train_err += train(inputs[0], inputs[1], inputs[2])
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            new_validlabels = []
            pred_validlabels = []
            # for batch in iterate_minibatches_listinputs([validAdmiSeqs, validLabels, validMask], N_BATCH, shuffle=True):
            #     inputs = batch
            #     err = compute_cost(inputs[0], inputs[1], inputs[2])
            #     new_validlabels.extend(inputs[1])
            #     pred_validlabels.extend(prd(inputs[0], inputs[2]))
            #     val_err += err
            #     val_batches += 1
            # val_auc = roc_auc_score(new_validlabels, pred_validlabels)
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation auc:\t\t{:.6f}".format(val_auc))
            # print("  validation accuracy:\t\t{:.2f} %".format(
            #     val_acc / val_batches * 100))

            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            new_testlabels = []
            pred_testlabels = []
            for batch in iterate_minibatches_listinputs([test_admiSeqs, test_labels, test_mask, testLengths], N_BATCH, shuffle=False):
                inputs = batch
                err = compute_cost(inputs[0], inputs[1], inputs[2])
                test_err += err
                leng = inputs[3][0]
                # new_testlabels.extend(inputs[1].flatten()[:leng])
                # pred_testlabels.extend(prd(inputs[0], inputs[2]).flatten()[:leng])
                new_testlabels.extend(cut_sel(inputs[1],inputs[3]))
                pred_testlabels.extend(cut_sel(inputs[0],inputs[3]))
                test_batches += 1
            test_auc = roc_auc_score(new_testlabels, pred_testlabels)
            np.save("CONTENT_results/testlabels_"+str(epoch),new_testlabels)
            np.save("CONTENT_results/predlabels_"+str(epoch),pred_testlabels)

            pred_binary = [x>0.5 for x in pred_testlabels]
            test_pre_rec_f1 = precision_recall_fscore_support(new_testlabels, pred_binary, average='binary')

            print("Final results:")
            print("  test loss:\t\t{:.6f}".format(test_err / test_batches))
            print("  test auc:\t\t{:.6f}".format(test_auc))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
            print("  test Precision, Recall and F1:\t\t{:.4f} %\t\t{:.4f}\t\t{:.4f}".format(test_pre_rec_f1[0], test_pre_rec_f1[1], test_pre_rec_f1[2]))

    except KeyboardInterrupt:
        pass


def prepare_data(seqs, labels, vocabsize, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    eventSeq = []

    for seq in seqs:
        t = []
        for visit in seq:
            t.extend(visit)
        eventSeq.append(t)
    eventLengths = [len(s) for s in eventSeq]


    if maxlen is not None:
        new_seqs = []
        new_lengths = []
        new_labels = []
        for l, s, la in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_lengths.append(l)
                new_labels.append(la)
            else:
                new_seqs.append(s[:maxlen])
                new_lengths.append(maxlen)
                new_labels.append(la[:maxlen])
        lengths = new_lengths
        seqs = new_seqs
        labels = new_labels

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen, vocabsize)).astype('int64')
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    y = np.zeros((n_samples, maxlen)).astype('int64')
    for idx, s in enumerate(seqs):
        x_mask[idx, :lengths[idx]] = 1
        for j, sj in enumerate(s):
            for tsj in sj:
                x[idx, j, tsj-1] = 1
    for idx, t in enumerate(labels):
        y[idx,:lengths[idx]] = t
        # if lengths[idx] < maxlen:
        #     y[idx,lengths[idx]:] = t[-1]

    return x, x_mask, y, lengths, eventLengths


def eval(epoch):
    new_testlabels = np.load("theta/testlabels_"+str(epoch)+".npy")
    pred_testlabels = np.load("theta/predlabels_"+str(epoch)+".npy")
    test_auc = roc_auc_score(new_testlabels, pred_testlabels)
    test_acc = accuracy_score(new_testlabels, pred_testlabels>0.5)
    print('AUC: %0.04f' % (test_auc))
    print('ACC: %0.04f' % (test_acc))
    pre, rec, threshold = precision_recall_curve(new_testlabels, pred_testlabels)
    test_pre_rec_f1 = precision_recall_fscore_support(new_testlabels, pred_testlabels > 0.5, average='binary')
    print("  test Precision, Recall and F1:\t\t{:.4f} %\t\t{:.4f}\t\t{:.4f}".format(test_pre_rec_f1[0],
                                                                                    test_pre_rec_f1[1],
                                                                                    test_pre_rec_f1[2]))
    epoch = 6
    rnn_testlabels = np.load("rnn_results/testlabels_" + str(epoch) + ".npy")
    rnn_pred_testlabels = np.load("rnn_results/predlabels_" + str(epoch) + ".npy")
    pre_rnn, rec_rnn, threshold_rnn = precision_recall_curve(rnn_testlabels, rnn_pred_testlabels)
    test_pre_rec_f1 = precision_recall_fscore_support(rnn_testlabels, rnn_pred_testlabels > 0.5, average='binary')
    test_auc = roc_auc_score(rnn_testlabels, rnn_pred_testlabels)
    test_acc = accuracy_score(rnn_testlabels, rnn_pred_testlabels > 0.5)
    print('rnnAUC: %0.04f' % (test_auc))
    print('rnnACC: %0.04f' % (test_acc))
    print("  rnn test Precision, Recall and F1:\t\t{:.4f} %\t\t{:.4f}\t\t{:.4f}".format(test_pre_rec_f1[0],
                                                                                    test_pre_rec_f1[1],
                                                                                    test_pre_rec_f1[2]))
    epoch = 6
    wv_testlabels = np.load("rnnwordvec_results/testlabels_" + str(epoch) + ".npy")
    wv_pred_testlabels = np.load("rnnwordvec_results/predlabels_" + str(epoch) + ".npy")
    pre_wv, rec_wv, threshold_wv = precision_recall_curve(wv_testlabels, wv_pred_testlabels)
    test_pre_rec_f1 = precision_recall_fscore_support(new_testlabels, wv_pred_testlabels > 0.5, average='binary')
    test_auc = roc_auc_score(wv_testlabels, wv_pred_testlabels)
    test_acc = accuracy_score(wv_testlabels, wv_pred_testlabels > 0.5)
    print('wvAUC: %0.04f' % (test_auc))
    print('wvACC: %0.04f' % (test_acc))
    print("  wv test Precision, Recall and F1:\t\t{:.4f} %\t\t{:.4f}\t\t{:.4f}".format(test_pre_rec_f1[0],
                                                                                    test_pre_rec_f1[1],
                                                                                    test_pre_rec_f1[2]))


    import matplotlib.pyplot as plt
    plt.plot(rec, pre, label='CONTENT')
    plt.plot(rec_rnn, pre_rnn, label='RNN')
    plt.plot(rec_wv, pre_wv, label='RNN+word2vec')
    plt.legend()

    plt.title("Precision-Recall Curves")


    plt.show()

def list2dic(_list):
    output = dict()
    for i in _list:
        if i in output:
            output[i] +=1
        else:
            output[i] = 0
    return output

def outputCodes(indexs, patientList):
    HightPat = []
    for i in indexs:
        HightPat.extend(patientList[i])
    high = list2dic(HightPat)
    items = sorted(high.items(), key=lambda d: d[1], reverse=True)
    for key, value in items[:20]:
        print(key,value)


def scatter(x, colors):
    import matplotlib.patheffects as PathEffects
    import seaborn as sns
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 50))
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    # We add the labels for each digit.
    txts = []
    for i in range(5):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    return f, ax, sc, txts

def clustering(thetaPath, dataset):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.manifold import TSNE

    thetas = np.asarray(np.load(thetaPath))
    ypred = MiniBatchKMeans(n_clusters=50).fit_predict(thetas).flatten()
    tsn = TSNE(random_state=2017,n_iter=300).fit_transform(thetas)
    scatter(tsn, ypred)
    plt.show()
    # indexs1 = np.where(ypred==1)[0]
    # indexs3 = np.where(ypred==3)[0]
    # indexs5 = np.where(ypred == 5)[0]
    # indexs9 = np.where(ypred == 9)[0]
    # indexs12 = np.where(ypred == 12)[0]
    #
    # X_test_data, Y_test_data = data_sets.get_data_from_type("test")
    # new_X = []
    # for s in X_test_data:
    #     ss = []
    #     for t in s:
    #         ss.extend(t)
    #     new_X.append(ss)
    # outputCodes(indexs1,new_X)
    # print("\n")
    #
    # outputCodes(indexs3, new_X)
    # print("\n")
    # outputCodes(indexs5, new_X)
    # print("\n")
    # outputCodes(indexs9, new_X)
    # print("\n")
    # outputCodes(indexs12, new_X)
    # print("\n")




if __name__ == '__main__':
    FLAGS = Config()
    data_sets = PatientReader(FLAGS)
    #main(data_sets)
    # thetaPath = "theta/thetas1.npy"
    # clustering(thetaPath,data_sets)
    eval(1)
