import torch
from torchvision import transforms
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from pipe_datasets import IDRiD_Dataset, IDRiD_Dataset_teacher, IDRiD_Dataset_unlabeled_preds
from pipe_dataset1 import IDRiD_Dataset
from pipe_dataset2 import IDRiD_Dataset_teacher
#from pipe_dataset3 import IDRiD_Dataset_unlabeled_preds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pipe_models import MTL


class Pipe():
    def __init__(self):
        # for all models
        rx = 450
        ry = 300
        old_x = 4288
        old_y = 2848
        self.Rx = rx / old_x
        self.Ry = ry / old_y
        self.M1 = MTL('M1').to(device)
        self.M2 = MTL('M2').to(device)
        #self.M3=MTL('M3').to(device)
        self.data_transformer = transforms.Compose([transforms.Resize((rx, ry)), transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])

    def fit_predict_M1(self, tasks, epochs):
        self.M1_train_ds = IDRiD_Dataset(self.data_transformer, 'train')
        self.M1_train_dl = DataLoader(self.M1_train_ds, batch_size=32, shuffle=True)
        self.M1_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.M1.parameters()),
                                            weight_decay=1e-6,
                                            momentum=0.9,
                                            lr=1e-3,
                                            nesterov=True)
        self.M1_scheduler = ReduceLROnPlateau(self.M1_optimizer,
                                              factor=0.5,
                                              patience=3,
                                              min_lr=1e-7,
                                              verbose=True)
        self.M1_criterion = nn.CrossEntropyLoss()
        self.M1.fit(self.M1_train_dl, self.M1_optimizer, self.M1_scheduler, self.M1_criterion, tasks, epochs, self.Rx,
                    self.Ry)
        self.M1.load_state_dict(torch.load("./M1_weights" + str(tasks) + ".pt"))
        self.M1.eval()
        data = pd.DataFrame()
        z = []
        for i, (imgs, retinopathy_label, macular_edema_label, fovea_center_labels, optical_disk_labels) in enumerate(
                self.M1_train_ds):
            one_row = self.M1.forward(imgs[None, :].to(device))
            z.append(torch.concat(one_row).detach().cpu().numpy())
        data = pd.DataFrame(z)
        data.to_csv('./drive/MyDrive/IDRID/Labels/train/M1_predictions' + str(tasks) + '.csv')

    def fit_predict_M2(self, tasks, epochs):
        self.M2_train_ds = IDRiD_Dataset_teacher(self.data_transformer, tasks, 'train')
        self.M2_train_dl = DataLoader(self.M2_train_ds, batch_size=32, shuffle=True)
        self.M2_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.M2.parameters()),
                                            weight_decay=1e-6,
                                            momentum=0.9,
                                            lr=1e-3,
                                            nesterov=True)
        self.M2_scheduler = ReduceLROnPlateau(self.M2_optimizer,
                                              factor=0.5,
                                              patience=3,
                                              min_lr=1e-7,
                                              verbose=True)
        self.M2_criterion = nn.KLDivLoss(reduction='batchmean')
        self.M2.fit(self.M2_train_dl, self.M2_optimizer, self.M2_scheduler, self.M2_criterion, tasks, epochs, self.Rx,
                    self.Ry)
        #data = pd.DataFrame()
        #z = []
        #for i, (imgs, retinopathy_label, macular_edema_label, fovea_center_labels, optical_disk_labels) in enumerate(
        #        self.M2_train_ds):
        #    one_row = self.M2.forward(imgs[None, :].to(device))
        #    z.append(torch.concat(one_row).detach().cpu().numpy())
        #data = pd.DataFrame(z)
        #data.to_csv('./drive/MyDrive/IDRID/Labels/train/M2_predictions' + str(tasks) + '.csv')

    def fit_M3(self, tasks, epochs):
        self.M3_train_ds = IDRiD_Dataset_unlabeled_preds(self.data_transformer, tasks, data_type='test')
        self.M3_train_dl = DataLoader(self.M3_train_ds, batch_size=32, shuffle=True)
        self.M3_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.M3.parameters()),
                                            weight_decay=1e-6,
                                            momentum=0.9,
                                            lr=1e-3,
                                            nesterov=True)
        self.M3_scheduler = ReduceLROnPlateau(self.M3_optimizer,
                                              factor=0.5,
                                              patience=3,
                                              min_lr=1e-7,
                                              verbose=True)
        self.M3_criterion = nn.KLDivLoss(reduction='batchmean')
        self.M3.fit(self.M3_train_dl, self.M3_optimizer, self.M3_scheduler, self.M3_criterion, tasks, epochs, self.Rx,
                    self.Ry)

    def fit_pipe(self, tasks, epochs):
        self.fit_predict_M1(tasks, epochs)
        self.fit_predict_M2(tasks, epochs)
        # self.fit_M3(tasks, epochs)
