import os
import joblib
import numpy as np
import random

from sklearn.model_selection import train_test_split
from custom_scaler import Custom_Scaler
from emc_data_multi_engine import Data_Box

class EmC_Order_Box:
    ''' Generate a sequence of orders in the form (ang_idx,dir_norm)
    
    Properties
    --------------------
    
        Number of orders/pattern/environment samples: n_order
        Dimension of pattern: ptrn_dim 
        Dimension of order = seq_len x 2 -> (angle index,value)
        Dimension of environment = env_dim
        
    Access
    ---------------------
        Pattern scaler: 
            self.ptrn_scaler
        Scaled values:
            Orders:  self.order
            Coresponding masked patterns: self.target_ptrn
            Environment: self.env
            BeamNumber: self.n_beam
        Splitted data:
            self.train_test_pairs
        
    '''
    
    def __init__(self,scaler_dir,env_dim = 0,norm_idx = False):
        self.ptrn_scaler = joblib.load(scaler_dir)
        self.env_dim = env_dim    
        self.norm_idx = norm_idx
    
    def generate_order_with_terget_ptrn(self):
        '''
        Given unscaled masked target pattern [-1,-1,... dir1, -1,...,dir2,-1...] in self.target_ptrn, scale it, and
        transform to scaled target pattern [nan,nan,... dir1_norm, nan,...,dir2_norm,nan...]
        and generate the order [[ang_idx_1,dir1_norm], [ang_idx_2,dir2_norm],...[ang_idx_1,dir1_norm],[ang_idx_2,dir2_norm]...]
        '''
        thre = -1e-2
        self.target_ptrn = np.where(self.target_ptrn < thre, np.nan, self.target_ptrn)
        self.target_ptrn = self.ptrn_scaler.transform(self.target_ptrn)
        
        self.order = []
        # loop i - all samples; loop j - find non-nan element and record in order_i
        # Then extend orders to seq_len and save to order_i_extend 
        # Then append orders_extend to order
        for i in range(self.n_order):
            order_i = []
            for j in range(self.ptrn_dim):
                if not (np.isnan(self.target_ptrn[i,j])):
                    if self.norm_idx: order_i.append([j/self.ptrn_dim,self.target_ptrn[i,j]])
                    else: order_i.append([j,self.target_ptrn[i,j]])
            assert len(order_i) > 0, f"No target for sample #{i}."
            order_i_extend = []
            pointer = 0
            assert len(order_i)<=self.seq_len, "Sequence length smaller than target beam numbers"
            while len(order_i_extend)<self.seq_len:
                order_i_extend.append(order_i[pointer])
                pointer = (pointer+1)%len(order_i)
            self.order.append(order_i_extend)        
        self.order = np.array(self.order)
    
#     def load_csv(self,target_dir,seq_len = 36):
#         self.seq_len = seq_len
#         self.target_ptrn = np.genfromtxt(target_dir,delimiter = ",")
#         (self.n_order,self.ptrn_dim) = self.target_ptrn.shape        
#         self.target_ptrn_org = np.copy(self.target_ptrn)
#         self.generate_order_with_terget_ptrn()
    
    def generate_rnd(self,ptrn_dim,sample_sizes,minmaxes,seq_len = 36,
                     ptrn_idx_range = None,
                     inc_split = 1,inc_range = None,
                     frq_split = 1,frq_range = None, seed = None,beam_null = None):
        '''
        Note: minmaxes are unsclaed range of pattern, env_range are scaled range of environment
        '''
        if seed is not None: random.seed(seed)
        
        self.ptrn_dim = ptrn_dim
        max_beam_number = len(sample_sizes)
        self.seq_len = seq_len
        self.target_ptrn = []
        self.n_beam = []
        
        if ptrn_idx_range is None:
            idx = list(range(self.ptrn_dim))
        else:
            idx = list(range(ptrn_idx_range[0],ptrn_idx_range[1]+1))
        self.inc = []
        if inc_range is None: inc_loop = range(inc_split)
        else: inc_loop = range(inc_range[0],inc_range[1]+1)
        for inc_idx in inc_loop:
            inc_ang = 1/((inc_split-1) if inc_split>1 else 1)*inc_idx
            # loop i: different beam numbers; loop k: samples for i beams; loop j: fill in pattern with beams
            for i in range(max_beam_number):
                beam = i+1
                size = sample_sizes[i]
                (minval,maxval) = minmaxes[i]
                for k in range(size):
                    idxes = random.sample(idx,beam)
                    this_ptrn = [-1 for i in range(self.ptrn_dim)]
                    for j in range(beam):
                        if beam_null is None:
                            val = random.uniform(minval,maxval)
                        elif j < beam_null:
                            val = maxval
                        else:
                            val = minval
                            
                        this_ptrn[idxes[j]] = val
                    self.target_ptrn.append(this_ptrn)
                    self.n_beam.append(beam)
                    self.inc.append(inc_ang)
                
        self.target_ptrn = np.array(self.target_ptrn)
        self.n_beam = np.array(self.n_beam)
        self.inc = np.array(self.inc)
        (self.n_order,self.ptrn_dim) = self.target_ptrn.shape        
        self.target_ptrn_org = np.copy(self.target_ptrn)
        self.generate_order_with_terget_ptrn()
        
        if frq_split>1:
            assert self.env_dim == 2
            if frq_range is None:
                self.frq = [random.randint(0,frq_split-1)/(frq_split-1) for i in range(self.n_order)]
            else:
                self.frq = [random.randint(frq_range[0],frq_range[1])/(frq_split-1) for i in range(self.n_order)]
            self.frq = np.array(self.frq)
            self.env = np.append(np.reshape(self.inc,(self.n_order,1)),
                                 np.reshape(self.frq,(self.n_order,1)),
                                 axis = -1)
        else:
            self.env = np.reshape(self.inc,(self.n_order,1))
            
#     def generate_rnd_from_data(self,emc_data_box,sample_portions,sample_size = 0,seq_len = 36):
#         # Fetch the pattern dimension and scaler directly from the data box
#         self.ptrn_dim = emc_data_box.data_dim()[2][1]
#         self.ptrn_scaler = emc_data_box.scalers[1]
#         self.seq_len = seq_len

#         # Fetch the data from data box to a list ptrn_from_box
#         if sample_size > emc_data_box.data_dim()[0]:
#             raise NameError("Not enough samples in data box")
#         if sample_size == 0 or sample_size == emc_data_box.data_dim()[0]:
#             sample_size = emc_data_box.data_dim()[0]
#             ptrn_from_box = emc_data_box.fetch_data(raw = True)[1].tolist()
#         else:
#             ptrn_from_box = emc_data_box.fetch_data(raw = True)[1].tolist()
#             ptrn_from_box = random.sample(ptrn_from_box,sample_size)
        
#         self.ptrn_from_box = ptrn_from_box
#         max_beam_number = len(sample_portions)
#         self.target_ptrn = []
#         sample_sizes = []
#         idx = list(range(self.ptrn_dim))
#         count = 0
        
#         # Divide samples
#         for i in range(max_beam_number-1):
#             sample_sizes.append(sample_size*sample_portions[i]//sum(sample_portions))
#         sample_sizes.append(sample_size-sum(sample_sizes))
        
#         # Generate datas: loop i - different beam numbers; loop k - sample size; look j - each beam
#         for i in range(max_beam_number):
#             beam = i+1
#             for k in range(sample_sizes[i]):
#                 idxes = random.sample(idx,beam)
#                 this_ptrn = [-1 for i in range(self.ptrn_dim)]
#                 for j in range(beam):
#                     this_ptrn[idxes[j]] = ptrn_from_box[count][idxes[j]]
#                 self.target_ptrn.append(this_ptrn)
#                 count += 1
                
#         self.target_ptrn = np.array(self.target_ptrn)
#         (self.n_order,self.ptrn_dim) = self.target_ptrn.shape        
#         self.target_ptrn_org = np.copy(self.target_ptrn)
#         self.generate_order_with_terget_ptrn()
#         print(f"number of samples generated: {sample_sizes}, totaling {self.n_order}, {self.ptrn_dim}")

    
    # Fetch with the orders and patterns splitting to train/test
    def split_orders(self,t_size,seed = 1,return_org_pattern = False,return_beam_number = False):
        if self.env_dim>0:
            data_to_split = [self.order,self.env,self.target_ptrn]
        else:
            data_to_split = [self.order,self.target_ptrn]
        if return_org_pattern:
            data_to_split.append(self.ptrn_from_box)
        if return_beam_number:
            data_to_split.append(self.n_beam)
            
        if t_size == 1:
            train_test_pairs = []
            for data in data_to_split:
                train_test_pairs.append([])
                train_test_pairs.append(np.array(data))
            self.train_test_pairs = tuple(train_test_pairs)
        else:
            self.train_test_pairs =  train_test_split(*data_to_split,test_size=t_size, random_state=seed)
        self.n_train = 0 if t_size == 1 else self.train_test_pairs[0].shape[0]
        self.n_test = self.train_test_pairs[1].shape[0]
        
        o_train_shape = 0 if t_size == 1 else self.train_test_pairs[0].shape
        o_test_shape = self.train_test_pairs[1].shape
        t_train_shape = 0 if t_size == 1 else self.train_test_pairs[4 if self.env_dim>0 else 2].shape
        t_test_shape = self.train_test_pairs[5 if self.env_dim>0 else 3].shape
        
        if self.env_dim>0:
            e_train_shape = 0 if t_size == 1 else self.train_test_pairs[2].shape
            e_test_shape = self.train_test_pairs[3].shape
        
        print(f"order_train shape: {o_train_shape}")
        print(f"order_test shape: {o_test_shape}")
        if self.env_dim>0:
            print(f"environment_train shape: {e_train_shape}")
            print(f"environment_test shape: {e_test_shape}")
        print(f"target_ptrn_train shape: {t_train_shape}")
        print(f"target_ptrn_test shape: {t_test_shape}")
        
        return  self.train_test_pairs
        

    def generate_instruct(self,ptrn_dim,instructions,envs = None, seq_len = 36):
        '''
        Instructions: [(thateind,targetdir) x target#] x sample, targetdir not scaled,
        Envs: scaled
        '''
        self.ptrn_dim = ptrn_dim
        self.seq_len = seq_len
        self.target_ptrn = []
        
        for inst in instructions:
            this_ptrn = [-1 for i in range(self.ptrn_dim)]
            for (tg_idx,tg_dir) in inst: this_ptrn[tg_idx] = tg_dir
            self.target_ptrn.append(this_ptrn)
        self.target_ptrn = np.array(self.target_ptrn)
        (self.n_order,self.ptrn_dim) = self.target_ptrn.shape        
        self.target_ptrn_org = np.copy(self.target_ptrn)
        
        self.generate_order_with_terget_ptrn()
        
        if self.env_dim>0:
            self.env = []
            for env in envs: self.env.append(env)
            self.env = np.array(self.env)
    
    
    def traintest_len(self):
        return (self.n_train,self.n_test)           
        
    # Import and Export
    def load(self,order_dir,target_dir,env_dir = None, ptrn_dir = None):
        self.order = joblib.load(order_dir)
        self.target_ptrn = joblib.load(target_dir)
        if env_dir != None:
            self.env = joblib.load(env_dir)
        if ptrn_dir != None:
            self.ptrn_from_box = joblib.load(ptrn_dir)

    def export(self,folder_dir,order_name,target_name,env_name = None,ptrn_name = None):
        if folder_dir != "":
            os.makedirs(folder_dir,exist_ok=True)
        joblib.dump(self.order, folder_dir+"/"+order_name)
        joblib.dump(self.target_ptrn, folder_dir+"/"+target_name)
        if env_name != None:
            joblib.dump(self.env, folder_dir+"/"+env_name)
        if ptrn_name != None:
            joblib.dump(self.ptrn_from_box, folder_dir+"/"+ptrn_name)