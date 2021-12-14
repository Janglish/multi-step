import os
import time
import torch
import pickle
import pandas as pd
from collections import namedtuple

class Logger:
    
    def __init__(self, dir_path="logs", file_name="logs", seed=None):
        self.dir_path = dir_path
        self.file_name = file_name
        self.start_time = time.time()
        os.makedirs(self.dir_path, exist_ok=True)
        self.df = pd.DataFrame()
        
        torch.manual_seed(seed)
    
    def write(self, reward):
        total_time = time.time() - self.start_time
        df = pd.DataFrame({'reward' : [reward], 'total_time' : [total_time]})
        self.df = self.df.append(df)
        self.df.to_csv(self.dir_path + '/' + self.file_name + '.csv', index=False)
        
    def save(self, model, path_or_buf=None):
        with open(path_or_buf, "wb") as f:
            pickle.dump(model, f)
        
    def load(self, path_or_buf=None):
        with open(path_or_buf, "rb") as f:
            model = pickle.load(f)
        model.eval()
        return model