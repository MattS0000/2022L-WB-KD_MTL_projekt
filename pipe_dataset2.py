from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
path2img = './drive/MyDrive/IDRID/Images/'
path2labels = './drive/MyDrive/IDRID/Labels/'

class IDRiD_Dataset_teacher(Dataset):
    def __init__(self, transform, tasks, data_type="train"):
        #path to images
        path2data = os.path.join(path2img,data_type)
        #list of images
        filenames = os.listdir(path2data)
        #fullpath
        self.full_filenames = [os.path.join(path2data,f) for f in filenames]
        #labels
        csv_filename="M1_predictions"+str(tasks)+".csv"
        path2csvLabels = os.path.join(path2labels,data_type,csv_filename)
        print(path2csvLabels)
        labels_df = pd.read_csv(path2csvLabels,index_col=[0])
        self.labels = labels_df.iloc[:,:]
        self.transform = transform
    
    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self,idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        table = self.labels.loc[idx].to_numpy()
        return image, table[0:5],table[5:10],table[10:12],table[12:]