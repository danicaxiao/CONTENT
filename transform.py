import numpy as np
import pandas as pd
from util import *

from config import Config
from patient_data_reader import PatientReader


input_file = '../resource/snow_case_with_facilities.txt'
vocab_file = '../resource/vocab.txt'
stop_file = '../resource/stop.txt'
vocab_pkl = '../resource/vocab.pkl'

rare_word = 100
stop_word = 1e4
unknown = 1


def dump_vocab():
    df = pd.read_csv(input_file, sep='\t', header=0)
    print(df[0:3])

    # .to_frame(): indexed by the groups, with a custom name
    # .reset_index(): set the groups to be columns again
    hist = df.groupby('DX_GROUP_DESCRIPTION').size().to_frame('SIZE').reset_index()
    print(hist[0:3])

    # show some stats
    hist_sort = hist.sort_values(by='SIZE', ascending=False)
    print(hist_sort[0:3])
    count = hist.groupby('SIZE').size().to_frame('COUNT').reset_index()
    print(count)

    # filter
    hist = hist[hist['SIZE'] > rare_word]
    print(hist)

    # dump
    vocab = hist.sort_values(by='SIZE').reset_index()['DX_GROUP_DESCRIPTION']
    vocab.index += 2  # reserve 1 to unk
    vocab.to_csv(vocab_file, sep='\t', header=False, index=True)

    # stop word
    hist[hist['SIZE'] > stop_word].reset_index()['DX_GROUP_DESCRIPTION']\
        .to_csv(stop_file, sep='\t', header=False, index=False)


def load_vocab():
    word_to_index = {}
    with open(vocab_file, mode='r') as f:
        line = f.readline()
        while line != '':
            tokens = line.strip().split('\t')
            word_to_index[tokens[1]] = int(tokens[0])
            line = f.readline()
    print('dict size: ' + str(len(word_to_index)))
    save_pkl(vocab_pkl, {v: k for k, v in word_to_index.items()})
    return word_to_index


def convert_format(word_to_index, events):
    # order by PID, DAY_ID
    with open(input_file, mode='r') as f:
        # header
        header = f.readline().strip().split('\t')
        print(header)
        pos = {}
        for key, value in enumerate(header):
            pos[value] = key
        print(pos)

        docs = []
        doc = []
        sent = []
        labels = []
        label = []

        # init
        line = f.readline()
        tokens = line.strip().split('\t')
        pid = tokens[pos['PID']]
        day_id = tokens[pos['DAY_ID']]
        label.append(tag(events, pid, day_id))

        while line != '':
            tokens = line.strip().split('\t')
            c_pid = tokens[pos['PID']]
            c_day_id = tokens[pos['DAY_ID']]

            # closure
            if c_pid != pid:
                doc.append(sent)
                docs.append(doc)
                sent = []
                doc = []
                pid = c_pid
                day_id = c_day_id
                labels.append(label)
                label = [tag(events, pid, day_id)]
            else:
                if c_day_id != day_id:
                    doc.append(sent)
                    sent = []
                    day_id = c_day_id
                    label.append(tag(events, pid, day_id))

            word = tokens[pos['DX_GROUP_DESCRIPTION']]
            try:
                sent.append(word_to_index[word])
            except KeyError:
                sent.append(unknown)

            line = f.readline()

        # closure
        doc.append(sent)
        docs.append(doc)
        labels.append(label)

    return docs, labels


def split_data(docs, labels):
    # train, validate, test
    # X, Y,
    # TODO: YY
    print(len(docs))
    #print(docs)
    print(len(labels))
    #print(labels)

    save_pkl('../resource/X_train.pkl', docs[:4000])
    save_pkl('../resource/Y_train.pkl', labels[:4000])
    save_pkl('../resource/X_valid.pkl', docs[4000:4700])
    save_pkl('../resource/Y_valid.pkl', labels[4000:4700])
    save_pkl('../resource/X_test.pkl', docs[4700:])
    save_pkl('../resource/Y_test.pkl', labels[4700:])


def extract_events():
    # extract event "INPATIENT HOSPITAL"
    target_event = 'INPATIENT HOSPITAL'

    df = pd.read_csv(input_file, sep='\t', header=0)
    events = df[df['SERVICE_LOCATION'] == target_event]

    events = events.groupby(['PID', 'DAY_ID', 'SERVICE_LOCATION']).size().to_frame('COUNT').reset_index()\
        .sort_values(by=['PID', 'DAY_ID'], ascending=True)\
        .set_index('PID')

    return events


def tag(events, pid, day_id):
    return 1 if tag_logic(events, pid, day_id) else 0


def tag_logic(events, pid, day_id):
    try:
        patient = events.loc[int(pid)]

        # test whether have events within 30 days
        if isinstance(patient, pd.Series):
            return (int(day_id) <= patient.DAY_ID) & (patient.DAY_ID < int(day_id) + 30)

        return patient.loc[(int(day_id) <= patient.DAY_ID) & (patient.DAY_ID < int(day_id) + 30)].shape[0] > 0
    except KeyError:
        # the label is not in the [index]
        return False


def main():
    # dump_vocab()
    word_to_index = load_vocab()
    events = extract_events()

    docs, labels = convert_format(word_to_index, events)
    split_data(docs, labels)

    # verify loading
    config = Config()
    reader = PatientReader(config)
    iterator = reader.iterator()
    X, Xc = iterator[0].next()
    Y, seq_len = iterator[1].next()
    #printing stuff to debug
    print("X is of shape CxT_patient: ", X)
    print("Xc is of shape Cx1: ", Xc.shape)  
    print("seq_len is of shape Cx1: ", seq_len.shape)

if __name__ == '__main__':
    main()
