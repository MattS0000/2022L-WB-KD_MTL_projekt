from pickle import TRUE
import torchvision.models as models
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="")


class MTL(nn.Module):
    def __init__(self, state="M1"):
        super(MTL, self).__init__()
        self.state = state
        resnet50 = models.resnet50(pretrained=True)
        self.features = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
        for param in self.features.parameters():
            param.requires_grad = False
        self.last = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
        self.retinopathy_classifier = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 5))
        self.macular_edema_classifier = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 5))
        self.fovea_center_cords = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 2))
        self.optical_disk_cords = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 2))
        if self.state == 'M2' or self.state == 'M3':
            self.float()

    def forward(self, data):
        out = self.features.forward(data).squeeze()
        out = self.last.forward(out)
        return (self.retinopathy_classifier(out),
                self.macular_edema_classifier(out),
                self.fovea_center_cords(out),
                self.optical_disk_cords(out))

    def fit(self, train_dl, optimizer, scheduler, criterion, tasks, epochs, Rx, Ry):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.criterion2 = nn.MSELoss()
        self.tasks = tasks
        self.epochs = epochs
        best_loss = float("Inf")
        for e in range(self.epochs):
            self.train()
            print("Epoch: {} ".format(e + 1), end="\n")
            current_loss = self.fit_iter(train_dl, Rx, Ry)
            if current_loss.item() < best_loss:
                best_loss = current_loss.item()
                #if self.state == 'M1':
                #    torch.save(self.state_dict(), 'M1_weights' + str(tasks) + '.pt')
                #elif self.state == 'M2' or self.state == 'M3':
                torch.save(self.state_dict(), self.state+'_weights' + str(tasks) + '.pt')
                print("Saved best model weights!")

    def fit_iter(self, train_dl, Rx, Ry):
        progress_bar(0, len(train_dl))
        train_loss = 0.0
        loss0sum = 0.0
        loss1sum = 0.0
        loss2sum = 0.0
        loss3sum = 0.0
        loss = torch.tensor(0)
        accuracy0 = 0.0
        accuracy1 = 0.0
        for i, tupel_train in enumerate(train_dl):
            if self.state == 'M1' or self.state == 'M3':
                imgs, retinopathy_label, macular_edema_label, fovea_center_labels, optical_disk_labels = tupel_train
            elif self.state == "M2":
                imgs, retinopathy_label, macular_edema_label, fovea_center_labels, optical_disk_labels, retinopathy_label2, macular_edema_label2, fovea_center_labels2, optical_disk_labels2 = tupel_train
            #imgs, retinopathy_label, macular_edema_label, fovea_center_labels, optical_disk_labels = tupel_train

            fovea_center_labels[:0], fovea_center_labels[:1] = fovea_center_labels[:0] * Rx, fovea_center_labels[:1] * Ry
            optical_disk_labels[:0], optical_disk_labels[:1] = optical_disk_labels[:0] * Rx, optical_disk_labels[:1] * Ry
            batch_size = imgs.size(0)
            self.optimizer.zero_grad()
            retinopathy_pred, macular_edema_pred, fovea_center_pred, optical_disk_pred = self.forward(imgs.to(device))
            if self.state == 'M1':
                loss0 = self.criterion(retinopathy_pred, retinopathy_label.to(device).to(torch.int64)).to(
                    torch.float64) * 10
                loss1 = self.criterion(macular_edema_pred, macular_edema_label.to(device).to(torch.int64)).to(
                    torch.float64) * 10
                loss2 = torch.sqrt(self.criterion2(fovea_center_pred.to(torch.double),fovea_center_labels.to(device).to(torch.double))) / 10
                loss3 = torch.sqrt(self.criterion2(optical_disk_pred.to(torch.double), optical_disk_labels.to(device).to(torch.double))) / 10
            elif self.state == 'M2':
                lossCE = nn.CrossEntropyLoss()
                lossKL = nn.KLDivLoss(reduction='batchmean')
                loss0CE = lossCE(retinopathy_pred, torch.reshape(retinopathy_label2, (-1,)).to(device).to(torch.int64)).to(torch.float64)
                loss1CE = lossCE(macular_edema_pred, torch.reshape(macular_edema_label2, (-1, )).to(device).to(torch.int64)).to(torch.float64)
                loss0KL = lossKL(F.log_softmax(retinopathy_pred.double(), dim=-1), retinopathy_label.to(device)) * 10
                loss1KL = lossKL(F.log_softmax(macular_edema_pred.double(), dim=-1), macular_edema_label.to(device)) * 10
                loss2 = torch.sqrt(self.criterion2(fovea_center_pred.to(torch.double), fovea_center_labels.to(device).to(torch.double)))
                loss3 = torch.sqrt(self.criterion2(optical_disk_pred.to(torch.double), optical_disk_labels.to(device).to(torch.double)))
                loss0 = loss0CE*0.6 + loss0KL*0.4
                loss1 = loss1CE*0.6 + loss1KL*0.4
            elif self.state =="M3":
                loss0 = self.criterion(F.log_softmax(retinopathy_pred.double(), dim=-1),
                                       F.softmax(retinopathy_label.to(device).double(), dim=-1)).double()
                loss1 = self.criterion(F.log_softmax(macular_edema_pred.double(), dim=-1),
                                       F.softmax(macular_edema_label.to(device).double(), dim=-1)).double()
                loss2 = torch.sqrt(self.criterion2(fovea_center_pred.to(torch.double),
                                                   fovea_center_labels.to(device).to(torch.double)))
                loss3 = torch.sqrt(self.criterion2(optical_disk_pred.to(torch.double),
                                                   optical_disk_labels.to(device).to(torch.double)))

            loss0sum += loss0 
            loss1sum += loss1
            loss2sum += loss2
            loss3sum += loss3
            if self.state == 'M1':
                pred0 = F.softmax(retinopathy_pred, dim=-1).argmax(dim=-1)
                accuracy0 += pred0.eq(retinopathy_label.to(device)).sum().item()
                pred1 = F.softmax(macular_edema_pred, dim=-1).argmax(dim=-1)
                accuracy1 += pred1.eq(macular_edema_label.to(device)).sum().item()
            loss = torch.stack((loss0, loss1, loss2, loss3))[self.tasks].sum()
# 
  #          lossKL = torch.stack((lossKL0, lossKL1, lossKL2, lossKL3))[self.tasks].sum()
            progress_bar(i + 1, len(train_dl))
            # print('Batch number: {}\nLoss on batch: {}\nLoss0: {}\nLoss1: {}\nLoss2: {}\nLoss3: {}\n-----------------------'.format(i,loss.item(), loss0.item(), loss1.item(), loss2.item() ,loss3.item()))
            loss.backward()
            self.optimizer.step()
            train_loss += loss
        if self.state == 'M1':
            print("\nTotal Loss: {}\nLoss0: {}  Accuracy0: {}\nLoss1: {}  Accuracy1: {}\nLoss2: {}\nLoss3: {}".format(
                train_loss, loss0sum, accuracy0 * 100 / 413, loss1sum, accuracy1 * 100 / 413, loss2sum, loss3sum))
        if self.state == 'M2' or self.state == 'M3':
            print(
                "\nTotal Loss: {}\nLoss0: {} \nLoss1: {}  \nLoss2: {}\nLoss3: {}".format(train_loss, loss0sum, loss1sum,
                                                                                         loss2sum, loss3sum))
        return train_loss
