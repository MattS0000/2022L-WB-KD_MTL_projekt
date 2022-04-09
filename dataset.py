from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
from PIL import Image
path2img='./IDRID/Images/'
path2labels='./IDRID/Labels/'

class IDRiD_Dataset(Dataset):
    def __init__(self, transform, data_type="train"):
        #path to images
        path2data=os.path.join(path2img,data_type)
        #list of images
        filenames=os.listdir(path2data)
        #fullpath
        self.full_filenames=[os.path.join(path2data,f) for f in filenames]
        #labels
        csv_filename="labels.csv"
        path2csvLabels=os.path.join(path2labels,data_type,csv_filename)
        labels_df=pd.read_csv(path2csvLabels,index_col=[0])
        self.labels=labels_df.iloc[:,1:]
        self.transform=transform
    
    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self,idx):
        image=Image.open(self.full_filenames[idx])
        image=self.transform(image)
        return image, self.labels.loc[idx].to_numpy()