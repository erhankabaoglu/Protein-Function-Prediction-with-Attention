import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

# data = torch.load("values.pth")

labels = {'GO:0000287': 0, 'GO:0000977': 1, 'GO:0000978': 2, 'GO:0001077': 3, 'GO:0001078': 4, 'GO:0003677': 5,
          'GO:0003682': 6, 'GO:0003690': 7, 'GO:0003697': 8, 'GO:0003700': 9, 'GO:0003714': 10, 'GO:0003723': 11,
          'GO:0003729': 12, 'GO:0003735': 13, 'GO:0003779': 14, 'GO:0003924': 15, 'GO:0004252': 16, 'GO:0004672': 17,
          'GO:0004674': 18, 'GO:0004842': 19, 'GO:0004872': 20, 'GO:0004930': 21, 'GO:0005096': 22, 'GO:0005102': 23,
          'GO:0005507': 24, 'GO:0005509': 25, 'GO:0005516': 26, 'GO:0005524': 27, 'GO:0005525': 28, 'GO:0008017': 29,
          'GO:0008022': 30, 'GO:0008134': 31, 'GO:0008233': 32, 'GO:0008270': 33, 'GO:0016887': 34, 'GO:0019899': 35,
          'GO:0019901': 36, 'GO:0019904': 37, 'GO:0020037': 38, 'GO:0030145': 39, 'GO:0031625': 40, 'GO:0032403': 41,
          'GO:0042803': 42, 'GO:0043565': 43, 'GO:0044212': 44, 'GO:0046982': 45, 'GO:0051015': 46, 'GO:0051082': 47,
          'GO:0061630': 48, 'GO:0098641': 49}


def plotData(train, val, type):
    d1 = {"Epoch": np.array([i + 1 for i in range(len(train))]), type: np.array(train)}
    d2 = {"Epoch": np.array([i + 1 for i in range(len(val))]), type: np.array(val)}
    d_f1 = pd.DataFrame(d1)
    d_f2 = pd.DataFrame(d2)

    sns.set(style='darkgrid')
    sns.lineplot(x='Epoch', y=type, data=d_f1)
    sns.lineplot(x='Epoch', y=type, data=d_f2)
    plt.savefig(type + ".jpg")
    plt.show()


"""data = torch.load("values.pth")

data['accuracies']['train'] = [i.cpu().numpy() for i in data['accuracies']['train']]
data['accuracies']['val'] = [i.cpu().numpy() for i in data['accuracies']['val']]
plotData(data['accuracies']['train'], data['accuracies']['val'], 'Accuracy')
plotData(data['losses']['train'], data['losses']['val'], 'Loss')"""


def plotConfMatrix(y_true, y_pred):
    c_m = confusion_matrix(y_true, y_pred)
    print(c_m)
    df_cm = pd.DataFrame(c_m, index=[i for i in labels.keys()],
                         columns=[i for i in labels.keys()])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.show()


def plotRucCurve(y_true, y_score, n):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=n)
    roc_auc = auc(fpr, tpr)
    print(fpr, tpr, _)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.savefig('roc_curve.jpg')
    plt.show()


def metrics_(y_true, y_pred):
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average="micro"), recall_score(y_true,
                                                                                                          y_pred,
                                                                                                          average="micro"), f1_score(
        y_true, y_pred, average="micro"), matthews_corrcoef(y_true, y_pred)


def readMetrics():
    labels_ = list(labels.keys())
    pr = np.load("metrics/pr.npy")
    rc = np.load("metrics/rc.npy")
    f1 = np.load("metrics/f1.npy")

    with open("metrics.txt", "w") as m:
        m.writelines(str(labels_) + "\n")
        m.writelines(str(pr.tolist()) + "\n")
        m.writelines(str(rc.tolist()) + "\n")
        m.writelines(str(f1.tolist()) + "\n")


"""y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

print(metrics_(y_true, y_pred))

y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.8, 0.6, 0.6, 0.2])
plotRucCurve(y_true, y_pred, 1)"""
