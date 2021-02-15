import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from os import listdir
from os.path import isfile, join

vocab = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
         'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'U': 18, 'V': 19, 'W': 20, 'X': 21, 'Y': 22, 'Z': 23, 'UNK': 24}

labels = {'GO:0000287': 0, 'GO:0000977': 1, 'GO:0000978': 2, 'GO:0001077': 3, 'GO:0001078': 4, 'GO:0003677': 5,
          'GO:0003682': 6, 'GO:0003690': 7, 'GO:0003697': 8, 'GO:0003700': 9, 'GO:0003714': 10, 'GO:0003723': 11,
          'GO:0003729': 12, 'GO:0003735': 13, 'GO:0003779': 14, 'GO:0003924': 15, 'GO:0004252': 16, 'GO:0004672': 17,
          'GO:0004674': 18, 'GO:0004842': 19, 'GO:0004872': 20, 'GO:0004930': 21, 'GO:0005096': 22, 'GO:0005102': 23,
          'GO:0005507': 24, 'GO:0005509': 25, 'GO:0005516': 26, 'GO:0005524': 27, 'GO:0005525': 28, 'GO:0008017': 29,
          'GO:0008022': 30, 'GO:0008134': 31, 'GO:0008233': 32, 'GO:0008270': 33, 'GO:0016887': 34, 'GO:0019899': 35,
          'GO:0019901': 36, 'GO:0019904': 37, 'GO:0020037': 38, 'GO:0030145': 39, 'GO:0031625': 40, 'GO:0032403': 41,
          'GO:0042803': 42, 'GO:0043565': 43, 'GO:0044212': 44, 'GO:0046982': 45, 'GO:0051015': 46, 'GO:0051082': 47,
          'GO:0061630': 48, 'GO:0098641': 49}


class SequenceDataset(Dataset):
    def __init__(self, fasta_file, label_file):
        dictarr = np.asarray(list(vocab.values())).reshape(-1, 1)
        self.enc = OneHotEncoder()
        self.enc.fit(dictarr)

        self.labels = self.parseLabel(label_file)
        self.sequences = {i.id: str(i.seq) for i in SeqIO.parse(fasta_file, "fasta")}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        id = list(self.sequences.keys())[item]
        sequence = list(self.sequences.values())[item]
        sequence = [[vocab[i]] if i in vocab.keys() else [vocab['UNK']] for i in list(sequence)]
        f_input = torch.tensor(self.enc.transform(sequence).toarray(), dtype=torch.float)
        label = torch.tensor(labels[self.labels[id]], dtype=torch.long)
        return f_input, label

    def parseLabel(self, label_file):
        with open(label_file, "r") as f:
            labels = {i.split()[0]: i.split()[1] for i in f.readlines()}
        return labels

    def nClasses(self):
        return len(labels)


def selectClasses(label_file, new_label_file, n_class):
    with open(label_file, "r") as f, open("leafonly_MFO.txt", "r") as t:
        data = f.readlines()
        test = t.readlines()
        test_p = set([i.split()[1].replace("\n", "") for i in test])
        go = set([i.split()[1] for i in data if i.split()[2].replace("\n", "") == "F" and i.split()[1] in test_p])
        dct = {i: 0 for i in go}
        count = 0
        for i in data:
            s = i.split()
            if s[1] in go:
                dct[s[1]] += 1
        dct = dict(sorted(dct.items(), key=lambda item: item[1]))
        subset_dct = {}
        for i in dct.keys():
            if dct[i] >= 150 and count < n_class:
                subset_dct[i] = dct[i]
                count += 1
        subset_dct = dict(sorted(subset_dct.items(), key=lambda item: item[1]))
        new_data = []
        c_dst = {i: 0 for i in subset_dct}
        proteins = []
        for i in data:
            n_s = i.split()
            if n_s[1] in list(subset_dct.keys()):
                if c_dst[n_s[1]] < 100 and n_s[0] not in proteins:
                    new_data.append(i)
                    proteins.append(n_s[0])
                    c_dst[n_s[1]] += 1

    with open(new_label_file, "w") as f:
        f.writelines(new_data)


def adjustFasta(fasta_file, new_fasta_file, label_file):
    with open(label_file, "r") as f:
        data = f.readlines()
        p_names = set([i.split()[0] for i in data])
        s_records = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            if record.id in p_names:
                s_records.append(record)
        SeqIO.write(s_records, new_fasta_file, "fasta")


def trainValidationSplit(fasta_file, label_file):
    with open(label_file, "r") as f:
        data = f.readlines()
        X = np.array([i.split()[0] for i in data])
        y = np.array([i.split()[1] for i in data])
        sss = StratifiedShuffleSplit(test_size=0.1, random_state=0)
        for train_index, val_index in sss.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

        with open("CAFA3_training_data/cafa3_train.txt", "w") as t, open("CAFA3_training_data/cafa3_val.txt", "w") as v:
            for i, j in zip(X_train, y_train):
                t.write(i + " " + j + " F\n")
            for i, j in zip(X_val, y_val):
                v.write(i + " " + j + " F\n")

        train_records = []
        val_records = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            if record.id in X_train:
                train_records.append(record)
            if record.id in X_val:
                val_records.append(record)

        SeqIO.write(train_records, "CAFA3_training_data/cafa3_train.fasta", "fasta")
        SeqIO.write(val_records, "CAFA3_training_data/cafa3_val.fasta", "fasta")


def adjustTest(subset, test_file):
    with open(subset, "r") as f, open(test_file, "r") as t:
        s = f.readlines()
        labels = set([i.split()[1] for i in s])
        data = []
        test = t.readlines()
        for i in test:
            label = i.split()[1]
            if label in labels:
                data.append(i)

    with open(test_file, "w") as t:
        t.writelines(data)


def targetFasta(directory, test_file):
    onlyfiles = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    with open(test_file, "r") as f:
        data = f.readlines()
        names = set([i.split()[0] for i in data])
        seq = []
        for fastas in onlyfiles:
            fasta = SeqIO.parse(fastas, "fasta")
            for record in fasta:
                if record.id in names:
                    seq.append(record)

        SeqIO.write(seq, "test_fasta.fasta", "fasta")


# selectClasses("CAFA3_training_data/uniprot_sprot_exp.txt", "CAFA3_training_data/uniprot_sprot_exp_subset.txt", 50)
# adjustFasta("CAFA3_training_data/uniprot_sprot_exp.fasta", "CAFA3_training_data/uniprot_sprot_exp_subset.fasta", "CAFA3_training_data/uniprot_sprot_exp_subset.txt")
# trainValidationSplit("CAFA3_training_data/uniprot_sprot_exp_subset.fasta", "CAFA3_training_data/uniprot_sprot_exp_subset.txt")
# adjustTest("CAFA3_training_data/uniprot_sprot_exp_subset.txt", "leafonly_MFO.txt")
# targetFasta("CAFA3_targets/Target files", "leafonly_MFO.txt")
# SequenceDataset("CAFA3_training_data/cafa3_val.fasta", 'CAFA3_training_data/cafa3_val.txt')



