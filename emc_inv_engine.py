## Inverse network generator class

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

class EmC_Inv_Generator:
    """ RNN layers + FC layers

    Network Strucutre
    -----------------------
        Input - (seq_len x input_dim) (env_dim)
        RNN x rnn_lay_num - (seq_len x rnn_sizes[i]) for middle layers 
                            (rnn_sizes[-1]) for the last layer
        FC x fc_lay_num - (fc_sizes[i])
        Output - (out_dim + pad_dim) 

    Methodes
    ------------------------
        Setting up RNN layers: rnn_paras()
        Seeting up FC layers: fc_paras()
    """
    def __init__(self,in_dim,out_dim,env_dim = 0,
                 pad_dim = 0,pad_val = 0.5,seq_len = None, name = "inv",
                 batch_size = None):
        
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.env_dim = env_dim
        if self.env_dim > 0:
            assert seq_len != None, "If there is evironment input, seq_len must be defined"
        self.pad_dim = pad_dim
        self.pad_val = pad_val
        self.seq_len = seq_len
        self.rnn_lay_num = 2
        self.rnn_sizes = [24,48]
        self.rnn_actv = ["tanh","tanh"]
        self.rnn_regu = [0,0]
        self.fc_lay_num = 1
        self.fc_sizes = [1000]
        self.fc_actv = ["relu"]
        self.out_actv = "relu"
        self.fc_regu = [0]
        self.limit = True
        self.rnd_init = False
        self.batch_size = batch_size
        # self.name is for naming the layers
        self.name = name
    
    def set_list(self,list_obj,value,size):
        if isinstance(value,list):
            assert len(value) == size, "Size mismatch."
            list_obj[:] = value
        else:
            list_obj[:] = [value]*size
    
    
    def rnn_paras(self,layer_num,size,actvfun = "tanh",regu=0,rnd_init = False):
        self.rnn_lay_num = layer_num        
        self.set_list(self.rnn_sizes,size,layer_num)
        self.set_list(self.rnn_actv,actvfun,layer_num)
        self.set_list(self.rnn_regu,regu,layer_num)
        self.rnd_init = rnd_init
        if rnd_init:
            assert (self.batch_size is not None), "For random init state, batch size has to be specified."
            
    def fc_paras(self,layer_num,size,actvfun="relu",limit = True,regu=0):
        self.fc_lay_num = layer_num
        self.set_list(self.fc_sizes,size,layer_num)
        self.set_list(self.fc_actv,actvfun,layer_num)        
        self.limit = limit
        self.set_list(self.fc_regu,regu,layer_num)   
               
    def hyperpara_summary(self) -> str:
        summ = "------- Inv Network -------\n"
        summ += f"  Input dim {self.input_dim}\n"
        summ += f"  Environment dim {self.env_dim}\n"
        if self.batch_size is not None:
            summ += f"  Batch size {self.batch_size}\n"
        summ += f"  Output dim {self.output_dim}\n"
        summ += "\n"
        
        # RNN layers parameters:
        summ += f"RNN Layers:\n"
        summ += f"    Layer sizes: {self.rnn_sizes}\n"
        summ += f"    Regularizatons: {self.rnn_regu}\n"   
        summ += f"    Activations: {self.rnn_actv}\n"
        summ += f"    Use rnd initial states: {self.rnd_init}\n"
        summ += "\n"
        
        # FC layers parameters:
        summ += f"FC Layers:\n"
        summ += f"    Layer sizes: {self.fc_sizes}\n"
        summ += f"    Regularizatons: {self.fc_regu}\n"   
        summ += f"    Activations: {self.fc_actv}\n"
        summ += f"    Output activation: {self.out_actv}\n"
        summ += "    Output limiter " + ("on\n" if self.limit else "off\n")
        
        return summ
    
    def __str__(self):
        return self.hyperpara_summary()
        
    def create_EmC_Inv(self) -> tf.keras.Model:
        
        def padding_1D(tensor):
            pad_setting = tf.constant([[0, 0,], [0, self.pad_dim]])
            tensor_padded = tf.pad(tensor, pad_setting,
                                   "CONSTANT",constant_values = self.pad_val)
            return tensor_padded
        
        if self.batch_size is not None:
            input_order = layers.Input(batch_input_shape=(self.batch_size,self.seq_len,self.input_dim),name = "io0_odr")
        else:
            input_order = layers.Input(shape=(self.seq_len,self.input_dim),name = "io0_odr")
        rnn = input_order
        
        if self.env_dim>0:
            input_env = layers.Input(shape = (self.env_dim,),name = "io0_env")
            env_reshape = layers.Reshape((1,self.env_dim),
                                         name = "io0_env_rshpe")(input_env)
#             env_repeat = layers.Lambda(lambda x: tf.repeat(x,
#                                                            repeats=[self.seq_len], 
#                                                            axis=-2),
#                                        name = "io0_env_repeat")(env_reshape) # (seq_len,env_dim)
            env_repeat = tf.repeat(env_reshape,repeats=[self.seq_len], axis=-2)
            input_concat = layers.Concatenate(axis = -1,
                                              name = "io0_concat")([input_order,env_repeat])
            rnn = input_concat
            
        # RNN layers
        for i in range(self.rnn_lay_num):
            layername = f"{self.name}_rnn{i}"
            rnn = layers.SimpleRNN(self.rnn_sizes[i],activation = self.rnn_actv[i],
                                   name = layername,
                                   kernel_regularizer=regularizers.L2(self.rnn_regu[i]),
                                   return_sequences = (False if i==self.rnn_lay_num-1 else True),
                                   stateful = self.rnd_init)(rnn)
                                   # Use the last state in last batch as initial state, equavalently randomize the initial state

        # FC layers
        fc = rnn
        for i in range(self.fc_lay_num):
            layername = f"{self.name}_fc{i}"
            fc = layers.Dense(self.fc_sizes[i],activation = self.fc_actv[i],
                              kernel_regularizer=regularizers.L2(self.fc_regu[i]),
                              name = layername)(fc)
        output_imp = layers.Dense(self.output_dim,activation = self.out_actv,
                                  name = "io1_imp_inv")(fc)
        
        # Limit the value to [0,1]
        if self.limit:
            output_imp = layers.ReLU(max_value = 1, name = "io1_imp_lim")(output_imp)
        # Padding the output        
        if self.pad_dim >0 :
            output_imp = layers.Lambda(lambda x: padding_1D(x),
                                       name = "io1_imp_pad")(output_imp)

        if self.env_dim>0:
            model = Model(inputs = [input_order,input_env],outputs = output_imp)
        else:
            model = Model(inputs = input_order, outputs = output_imp)
        return model 