import os
import sys
import numpy as np

def read_all_sentences():
    path = os.getcwd()
    parent = os.path.dirname(path)
    data_dir = "data"
    data_dir = os.path.join(parent, data_dir)
    dataset_dir = "dataset"
    data_dir = os.path.join(data_dir, dataset_dir)
    files = os.listdir(data_dir)

    t = []
    X = []

# for each file, tokenize the sentences and add them to corpora.Dictionary
    for file in files:
        with open(os.path.join(data_dir, file), encoding='utf-8', mode='r') as f:
            line = f.readline()
            while line:
                basename = os.path.splitext(os.path.basename(file))[0]
                if basename[-4:] == "temp":
                    num = basename[-6]
                else:
                    num = basename[-1]

                sentence = line[:-3]
                line = f.readline()

                X.append(sentence)
                t.append([int(num)-1])

    n_labels = len(np.unique(t))
    #print("one-hot",np.eye(n_labels)[t].reshape(-1,5))
    return (np.array(X), np.eye(n_labels)[t].reshape(-1,5))
