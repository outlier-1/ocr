import numpy as np
import cv2
import os
from pathlib import Path

class DataSet:
    features = "feat"
    labels = "label"
    one_hot_labels = "onehot"

    def __init__(self, source_path):
        self.data_dict = self.checkPath(source_path)
        self.train_features = None
        self.train_labels = None
        self.train_labels_oh = None
        self.val_features = None
        self.val_labels = None
        self.val_labels_oh = None
        self.test_features = None
        self.test_labels = None
        self.test_labels_oh = None
        self.split_data_dict()

    def split_data_dict(self):
        for key in self.data_dict.keys():
            if "train" and "feat" in key:
                self.train_features = self.data_dict.get(key)
            elif "train" and "label" in key and "one" not in key:
                self.train_labels = self.data_dict.get(key)
            elif ("train" and "one") in key:
                self.train_labels_oh = self.data_dict.get(key)

            elif "val" and "feat" in key:
                self.val_features = self.data_dict.get(key)
            elif "val" and "label" in key and "one" not in key:
                self.val_labels = self.data_dict.get(key)
            elif "val" and "one" in key:
                self.val_labels_oh = self.data_dict.get(key)

            elif "test" and "feat" in key:
                self.test_features = self.data_dict.get(key)
            elif "test" and "label" in key and "one" not in key:
                self.test_labels = self.data_dict.get(key)
            elif "test" and "one" in key:
                self.test_labels_oh = self.data_dict.get(key)
        print("Data set Successfully Loaded.")

    def checkPath(self, path):
        try:
            dict = np.load(path).item()
            return dict
        except FileNotFoundError:
            print("File {} could not found.\nYour current directory:{}".format(path, Path.cwd()))
        except EOFError:
            print("Your file does not have .npy extension. Please check that again. ")

    # def get(self, ds: dict, mode):
    #     for key in ds.keys():
    #         if mode in key:
    #             return ds.get(key)

cl = DataSet(source_path='text_det.npy')
item = cl.get(cl.test,cl.one_hot_labels)




