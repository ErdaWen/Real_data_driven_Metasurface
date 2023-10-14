## Databox class
import os
import joblib
import numpy as np

from custom_scaler import Custom_Scaler
from sklearn.model_selection import train_test_split

class Data_Box:
    """ Store and process multiple features/labels
    
    Access
    --------------------------
    scaler: self.scalers
    original data: self.data_org
    scaled data: self.data
    """
    def __init__(self,data_dirs:[str],scales,
                 name = "", delim = None):        
        self.n_data = len(data_dirs)
        self.data = []
        self.data_org = []
        self.scalers = []
        self.dims = []
        self.n_samp = 0
        
        for i in range(self.n_data):
            self.scalers.append(Custom_Scaler(scales[i][0],scales[i][1]))
            data_i = np.genfromtxt(data_dirs[i],delimiter = delim)
            
            # for data with dimension of 1, we need to reshape it to (n_sample,1)
            if len(data_i.shape) == 1:
                data_i = np.reshape(data_i,(data_i.shape[0],1))
            
            samp_data_i,dim_data_i = data_i.shape
            assert self.n_samp == 0 or self.n_samp == samp_data_i, "inconsistant number of samples"
            self.n_samp = samp_data_i
            self.dims.append(dim_data_i)
            print(f"{data_dirs[i]} readed")
            self.data_org.append(data_i)
            data_i = self.scalers[i].transform(data_i)
            self.data.append(data_i)

        print(f"# of features/labels: {self.n_data}, # samples: {self.n_samp}")
        print(f"Dimensions: {self.dims}")
        
        if name == "":
            self.name = f"{self.n_data}_data_{self.dims}x{self.n_samp}]"
        else:
            self.name = data_name

    def data_dim(self):
        """ Get (number of data,number of samples, dim of features/labels)"""
        return (self.n_data, self.n_samp, self.dims)
        
    def export_scaler(self,folder_dir,scaler_names):
        if folder_dir != "":
            os.makedirs(folder_dir,exist_ok=True)
        for i in range(self.n_data):
            joblib.dump(self.scalers[i], f"{folder_dir}/{scaler_names[i]}")
            print(f"{folder_dir}/{scaler_names[i]} saved")
        
    def gen_train_test(self,t_size,seed = 1):
        data_pairs = train_test_split(*self.data,test_size=t_size, random_state=seed)
        self.n_train, self.n_test = data_pairs[0].shape[0],data_pairs[1].shape[0]
        print(f"Splitted (# of train , # of test): {self.train_test_len()}")
        return data_pairs
    
    def train_test_len(self):
        return (self.n_train,self.n_test)     