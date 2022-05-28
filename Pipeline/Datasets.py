from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

path2img = './drive/MyDrive/IDRID/Images/'
path2labels = './drive/MyDrive/IDRID/Labels/'


class IDRiD_General_Dataset(Dataset):
    def __init__(self, transform, data_type, csv_filename, index=1):
        # path to images
        path2data = os.path.join(path2img, data_type)
        # list of images
        filenames = os.listdir(path2data)
        # fullpath
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        # labels
        path2csvLabels = os.path.join(path2labels, data_type, csv_filename)
        print(path2csvLabels)
        labels_df = pd.read_csv(path2csvLabels, index_col=[0])
        self.labels = labels_df.iloc[:, index:]
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        return 'fake_item_super_class', 2, 3, 4, 'fake_item_super_class'


class IDRiD_Dataset(IDRiD_General_Dataset):
    def __init__(self, transform, data_type="train"):
        # path to images
        super().__init__(transform, data_type, "labels.csv", index=1)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        table = self.labels.loc[idx].to_numpy()
        return image, table[0], table[1], table[2:4], table[4:6]


class IDRiD_Dataset_Teacher(IDRiD_General_Dataset):
    def __init__(self, transform, tasks, data_type="train",  model_num = "M1", index=0):
        # path to images
        super().__init__(transform, data_type, model_num + "_predictions" + str(tasks) + ".csv", index=index)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        table = self.labels.loc[idx].to_numpy()
        return image, table[0:5], table[5:8], table[8:10], table[10:]


class IDRiD_Dataset_Unlabeled_Preds(IDRiD_Dataset_Teacher):
    def __init__(self, transform, tasks, data_type="train"):
        super().__init__(transform, tasks, data_type=data_type, model_num = "M2",  index=0)


class IDRiD_Dataset2(IDRiD_General_Dataset):
    def __init__(self, transform, tasks, csv_filename, data_type="train", model_num="M1", index=0):
        # path to images
        path2data = os.path.join(path2img, data_type)
        # list of images
        filenames = os.listdir(path2data)
        # fullpath
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        # labels
        path_to_preds = os.path.join(path2labels, data_type, 'Ensemble_predictions' + str(tasks) + '.csv')
        path2csvLabels = os.path.join(path2labels, data_type, "labels.csv")
        print(path2csvLabels)
        labels_df = pd.read_csv(path2csvLabels, index_col=[0])
        preds_df = pd.read_csv(path_to_preds, index_col = 0)
        merged_df = pd.concat([preds_df, labels_df], axis=1)
        self.labels = merged_df.drop('Image name', axis=1)
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        table = self.labels.loc[idx].to_numpy()
        return image, table[0:5], table[5:8], table[8:10], table[10:12], table[12:13], table[13:14], table[14:16], table[16:18]