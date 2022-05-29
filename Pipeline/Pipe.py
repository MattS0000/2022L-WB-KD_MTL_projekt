import torch
from torchvision import transforms
from itertools import combinations
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Pipeline.Datasets import IDRiD_Dataset, IDRiD_Dataset_Teacher, IDRiD_Dataset_Unlabeled_Preds, IDRiD_Dataset2
from Pipeline.Models import MTL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.M3 = MTL('M3').to(device)
        # ensemble dict stores (task1, task2, ...):Model
        self.ensemble = {}
        self.data_transformer = transforms.Compose([transforms.Resize((rx, ry)), transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])

    def read_predictions(self, filename):
        table = pd.read_csv(filename, index_col=0)
        r_label, m_label = table.iloc[:, 0:5], table.iloc[:, 5:8]
        r_label = softmax(torch.tensor(r_label.values), dim=-1)
        m_label = softmax(torch.tensor(m_label.values), dim=-1)
        f_coords, o_coords = table.iloc[:, 8:10], table.iloc[:, 10:]
        return r_label, m_label, f_coords, o_coords

    def fit_ensemble_submodel_predict(self, subtasks, epochs):
        self.ensemble[tuple(subtasks)] = MTL('M1').to(device)
        sub_train_ds = IDRiD_Dataset(self.data_transformer, 'train')
        sub_train_dl = DataLoader(sub_train_ds, batch_size=32, shuffle=True)
        sub_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.ensemble[tuple(subtasks)].parameters()),
                                        weight_decay=1e-6,
                                        momentum=0.9,
                                        lr=1e-3,
                                        nesterov=True)
        sub_scheduler = ReduceLROnPlateau(sub_optimizer,
                                          factor=0.5,
                                          patience=3,
                                          min_lr=1e-7,
                                          verbose=True)
        W_T0 = torch.tensor([0.53, 2.13, 0.53, 0.74, 1.06]).to(device)
        W_T1 = torch.tensor([0.6, 1.8, 0.6]).to(device)
        sub_criterion_t0 = nn.CrossEntropyLoss(weight = W_T0)
        sub_criterion_t1 = nn.CrossEntropyLoss(weight = W_T1)
        self.ensemble[tuple(subtasks)].fit(sub_train_dl, sub_optimizer, sub_scheduler, sub_criterion_t0, sub_criterion_t1,
                                            subtasks, epochs, self.Rx, self.Ry)
        self.ensemble[tuple(subtasks)].load_state_dict(torch.load("./drive/MyDrive/IDRID/Labels/train/M1_weights" + str(subtasks) + ".pt"))
        self.ensemble[tuple(subtasks)].eval()
        data = pd.DataFrame()
        z = []
        for i, (imgs, retinopathy_label, macular_edema_label, fovea_center_labels, optical_disk_labels) \
                in enumerate(sub_train_ds):
            one_row = self.ensemble[tuple(subtasks)].forward(imgs[None, :].to(device))
            z.append(torch.concat(one_row).detach().cpu().numpy())
        data = pd.DataFrame(z)
        data.to_csv('./drive/MyDrive/IDRID/Labels/train/M1_predictions' + str(subtasks) + '.csv')

    def fit_M1(self, tasks, epochs):
        all_combs = [list(i) for k in range(len(tasks)) for i in list(combinations(tasks, k + 1))]
        for sub_tasks in all_combs:
            print("Teaching submodel ", str(sub_tasks))
            self.fit_ensemble_submodel_predict(sub_tasks, epochs)

    def predict_M1(self, tasks, epochs):
        sub_data12 = self.read_predictions("./drive/MyDrive/IDRID/Labels/train/M1_predictions[0, 1].csv")
        sub_data1 = self.read_predictions("./drive/MyDrive/IDRID/Labels/train/M1_predictions[0].csv")
        sub_data2 = self.read_predictions("./drive/MyDrive/IDRID/Labels/train/M1_predictions[1].csv")
        joined_data = (sub_data1[0].numpy(),
                       pd.DataFrame(sub_data1[1].numpy(), columns=[5, 6, 7]), sub_data2[2], sub_data2[3])
        sub_data12 = (sub_data12[0].numpy(),
                      pd.DataFrame(sub_data12[1].numpy(), columns=[5, 6, 7]), sub_data12[2],sub_data12[3])
        voting = [pd.DataFrame(0.4 * x + 0.6 * y) for x, y in zip(sub_data12, joined_data)]
        prediction = pd.concat(voting, axis=1)
        prediction.to_csv('./drive/MyDrive/IDRID/Labels/train/Ensemble_predictions' + str(tasks) + '.csv')

    def fit_predict_M2(self, tasks, epochs):
        self.M2_train_ds = IDRiD_Dataset2(self.data_transformer, tasks, 'train')
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
        self.M2.fit(self.M2_train_dl, self.M2_optimizer, self.M2_scheduler, self.M2_criterion, self.M2_criterion, tasks, epochs, self.Rx, self.Ry)
        self.M2.load_state_dict(torch.load("./drive/MyDrive/IDRID/Labels/train/M2_weights" + str(tasks) + ".pt"))
        self.M2.eval()
        data = pd.DataFrame()
        z = []
        for i, (imgs, retinopathy_label, macular_edema_label, fovea_center_labels, optical_disk_labels, retinopathy_label2, macular_edema_label2, fovea_center_labels2, optical_disk_labels2) in enumerate( self.M2_train_ds):
            one_row = self.M2.forward(imgs[None, :].to(device))
            z.append(torch.concat(one_row).detach().cpu().numpy())
        data = pd.DataFrame(z)
        data.to_csv('./drive/MyDrive/IDRID/Labels/train/M2_predictions' + str(tasks) + '.csv')


    def fit_predict_M3(self, tasks, epochs):
        self.M3_train_ds = IDRiD_Dataset_Unlabeled_Preds(self.data_transformer, tasks, data_type="train")
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
        self.M3.fit(self.M3_train_dl, self.M3_optimizer, self.M3_scheduler, self.M3_criterion, self.M3_criterion, tasks, epochs, self.Rx,
                    self.Ry)
        self.M3.load_state_dict(torch.load("./drive/MyDrive/IDRID/Labels/train/M3_weights" + str(tasks) + ".pt"))
        self.M3.eval()
        self.M3_test_ds = IDRiD_Dataset(self.data_transformer, 'train')
        self.M3_test_dl = DataLoader(self.M3_test_ds, batch_size=32, shuffle=True)
        data = pd.DataFrame()
        z = []
        for i, (imgs, retinopathy_label, macular_edema_label, fovea_center_labels, optical_disk_labels) in enumerate(
                self.M3_test_ds):
            one_row = self.M3.forward(imgs[None, :].to(device))
            z.append(torch.concat(one_row).detach().cpu().numpy())
        data = pd.DataFrame(z)
        data.to_csv('./drive/MyDrive/IDRID/Labels/train/M3_predictions' + str(tasks) + '.csv')

    def fit_pipe(self, tasks, epochs):
        self.fit_M1(tasks, epochs)
        self.predict_M1(tasks, epochs)
        self.fit_predict_M2(tasks, epochs)
        self.fit_predict_M3(tasks, epochs)
