
from pandas.io import pickle
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
from sklearn.utils import shuffle
from glob import glob
import pickle
import matplotlib.pyplot as plt
import config

class CanineDataset(Dataset):
    def __init__(self, radiographs, masks, transform):
        self.__radiographs = radiographs;
        self.__masks = masks;
        self.__transform = transform;
            
    def __len__(self):
        return len(self.__radiographs);

    def __getitem__(self, index):
        radiograph_image_path = self.__radiographs[index];
        
        radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{radiograph_image_path}.jpeg'),cv2.IMREAD_GRAYSCALE);
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);

        mask =  pickle.load(open(self.__masks[index], 'rb'));

        transformed = self.__transform(image = radiograph_image, mask = mask);
        radiograph_image = transformed["image"];
        mask = transformed["mask"];
        return radiograph_image, mask;