import copy
import argparse
from model import PFPModel
from dataset import SequenceDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import datetime
import metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn_padd(batch):
    lengths = torch.tensor([t[1] for t in batch], dtype=torch.long)
    seq = [i[0] for i in batch]
    batch = torch.nn.utils.rnn.pad_sequence(seq)
    return batch, lengths


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train(opt):
    epochs, batch_size, resume = opt.epoch, opt.batch_size, opt.resume

    trainDataset = SequenceDataset("CAFA3_training_data/cafa3_train.fasta",
                                   "CAFA3_training_data/cafa3_train.txt")
    valDataset = SequenceDataset("CAFA3_training_data/cafa3_val.fasta",
                                 "CAFA3_training_data/cafa3_val.txt")

    dataset_sizes = {"train": len(trainDataset), "val": len(valDataset)}
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                 collate_fn=collate_fn_padd)
    valDataLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, num_workers=4,
                               collate_fn=collate_fn_padd)
    dataLoader = {"train": trainDataLoader, "val": valDataLoader}

    n_classes = trainDataset.nClasses()

    model = PFPModel(25, 256, n_classes, 2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs / 2, gamma=0.1)

    best_model_w = copy.deepcopy(model.state_dict())
    best_acc = 0
    losses = {"train": [], "val": []}
    accuracies = {"train": [], "val": []}

    epoch = 0

    if resume:
        checkpoint = torch.load("values.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        epochs += epoch
        losses = checkpoint['losses']
        accuracies = checkpoint['accuracies']

    for i in range(epoch, epochs):
        print('Epoch {}/{}'.format(i, epochs - 1))
        print('-' * 10)

        for phase in ["train", "val"]:
            running_loss = 0.0
            running_corrects = 0

            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, labels in dataLoader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    clip_gradient(model, 1e-1)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(1)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]) * 100
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('{} Loss: {:.4f} Acc: {:.4f} Time: {}'.format(
                phase, epoch_loss, epoch_acc, current_time))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_w = copy.deepcopy(model.state_dict())
        epoch = i

    torch.save({
        'epoch': epoch + 1,
        'losses': losses,
        'accuracies': accuracies,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': best_model_w
    }, "values.pth")


def test(opt):
    batch_size = opt.batch_size

    testDataset = SequenceDataset("test_fasta.fasta",
                                  "leafonly_MFO.txt")

    testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                collate_fn=collate_fn_padd)

    n_classes = testDataset.nClasses()
    values = torch.load("values.pth")

    model = PFPModel(25, 256, n_classes, 2)
    model.load_state_dict(values['model_state_dict'])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in testDataLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.append(_)

        acc, pr, rc, f1, mcc = metrics.metrics_(y_true, y_pred)
        print(acc, pr, rc, f1, mcc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='Number of epoch')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume')
    parser.add_argument('--test', nargs='?', const=True, default=False, help='test')
    opt = parser.parse_args()
    if opt.test:
        test(opt)
    else:
        train(opt)
