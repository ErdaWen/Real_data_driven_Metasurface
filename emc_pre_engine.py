## Predictor generator class

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

class EmC_Pre_Generator:
    """ CNN layers + FC layers

     Network Strucutre
     -----------------------
         Input - (input_dim) (env_dim)
         Reshape - (input_dim x 1) (env_dim x 1)
         Conv x conv_lay_num - (pre_layer_units/conv_clusters[i],channels[i])
         Flatten - (conv_clusters[conv_lay_num-1] x channels[conv_lay_num])
         FC x fc_la_num - (fc_sizes[i])
         Output - (out_dim)
         
         
    Methodes
    ------------------------
        Setting up Conv layers: conv_paras()
        Seeting up FC layers: fc_paras()
        Build a Pre model: create_EmC_Pre()
        Load (and lock) the weights: load_net()
    """
    
    def __init__(self,in_dim,out_dim, env_dim = 0,
                 name="pre",input_names = None,output_name = "io2_ptrn"):
        
        self.input_dim = in_dim
        self.env_dim = env_dim
        self.output_dim = out_dim
        
        # Environment FC layers parameters:
        self.fc_env_lay_num = 0
        self.fc_env_sizes = []
        self.fc_env_regu = []
        self.fc_env_actv = []
        
        # Convolutional layers parameters:
        self.conv_lay_num = 1
        self.conv_clusters = [2]
        self.conv_actv = ["relu"]
        self.conv_regu = [0]
        self.conv_channels = [72]
        self.env_on_all_conv = False
        
        # FC layers parameters:
        self.fc_regu = [0,0,0]
        self.fc_lay_num = 3
        self.fc_sizes = [1200,1200,1200]
        self.fc_actv = ["relu","relu","relu"]
        self.fc_regu = [0,0,0]
        self.out_actv = "relu"
        
        # self.name is for naming the layers
        self.name = name
        # if the Pre is used seperatly and a model is needed, include "_in" in the input_names 
        if input_names == None:
            if env_dim == 0:
                self.input_names = ["io1_imp_in"]
            else:
                self.input_names = ["io1_imp_in","io1_env_in"]
        elif isinstance(input_names,str):
            self.input_names = [input_names]
        else:
            self.input_names = input_names
        self.output_name = output_name
        
    
    def set_list(self,list_obj,value,size):
        if isinstance(value,list):
            assert len(value) == size, "Size mismatch."
            list_obj[:] = value
        else:
            list_obj[:] = [value]*size
    
    def conv_paras(self,layer_num,cluster,channel,
                   actvfun="relu",regu=0,env_on_all_conv = False):
        
        self.conv_lay_num = layer_num
        self.env_on_all_conv = env_on_all_conv
        
        self.set_list(self.conv_clusters,cluster,layer_num)
        self.set_list(self.conv_channels,channel,layer_num)
        self.set_list(self.conv_actv,actvfun,layer_num)
        self.set_list(self.conv_regu,regu,layer_num)

        # Validation:
        inforloss = False
        laysz = self.input_dim
        for c in self.conv_clusters:
            (laysz,m) = divmod(laysz,c)
            if m!=0:
                inforloss = True
                break
        if inforloss: print("Conv layer indivisible causing information Loss!")
    
    def fc_paras(self,layer_num,sz,actvfun="relu",regu=0):
        self.fc_lay_num = layer_num
        self.set_list(self.fc_sizes,sz,layer_num)
        self.set_list(self.fc_actv,actvfun,layer_num)
        self.set_list(self.fc_regu,regu,layer_num)
    
    def fc_env_paras(self,layer_num,sz,actvfun="relu",regu=0):
        self.fc_env_lay_num = layer_num
        self.set_list(self.fc_env_sizes,sz,layer_num)
        self.set_list(self.fc_env_actv,actvfun,layer_num)
        self.set_list(self.fc_env_regu,regu,layer_num)
    
    def hyperpara_summary(self) -> str:
        summ = "------- Pre Network -------\n"
        summ += f"  Input dim {self.input_dim}\n"
        summ += f"  Environment dim {self.env_dim}\n"
        summ += f"  Output dim {self.output_dim}\n"
        summ += "\n"
        
        # Environment FC layers parameters:
        if self.fc_env_lay_num > 0:
            summ += f"Environment FC:\n"
            summ += f"    Layer sizes: {self.fc_env_sizes}\n"
            summ += f"    Regularizatons: {self.fc_env_regu}\n"   
            summ += f"    Activations: {self.fc_env_actv}\n"
            summ += "\n"
        
        # Convolutional layers parameters:
        summ += f"Conv Layers:\n"
        summ += f"    Layer clusters: {self.conv_clusters}\n"
        summ += f"    Channels: {self.conv_channels} \n"
        summ += f"    Regularizatons: {self.conv_regu}\n"   
        summ += f"    Activations: {self.conv_actv}\n"
        summ += f"    Environment loaded on "
        summ += "all conv layers\n" if self.env_on_all_conv else "first conv layers\n"
        summ += "\n"
        
        # FC layers parameters:
        summ += f"FC Layers:\n"
        summ += f"    Layer sizes: {self.fc_sizes}\n"
        summ += f"    Regularizatons: {self.fc_regu}\n"   
        summ += f"    Activations: {self.fc_actv}\n"
        summ += f"    Output activation: {self.out_actv}\n"
        
        return summ
    
    def __str__(self):
        return self.hyperpara_summary()

    def create_EmC_Pre(self,prelayer = None, envlayer = None) -> (tf.keras.Model,layers.Dense):
        """ 
        Return (model,output_layer) if self.input_names[0] contains "in", indicating the first layer is input.
        Return (None,output_layer) if self.input_names[0] does not contain "in"
        , indication the first layer is an output of other network.
        """
        
        # Arrange the input as input_imp and [input_envs]
        if prelayer != None:
            input_imp = prelayer
            if envlayer != None:
                assert self.env_dim>0, "env_dim is not set"
                input_env = envlayer
        else:
            input_imp = layers.Input(shape=(self.input_dim,),name = self.input_names[0])
            if self.env_dim>0:
                input_env = layers.Input(shape=(self.env_dim,),name = self.input_names[1])
   
        # Reshape input_imp and input_env: (input_dim,1) 
        imp_reshape = layers.Reshape((self.input_dim,1),
                                     name = self.input_names[0]+"_"+self.name+"_rshpe")(input_imp)
        
        # FC for env (if fc_env_lay_num>0) and reshape to (env_dim_out,1)
        if self.env_dim>0:
            if self.fc_env_lay_num>0:
                fc_env = input_env
                for i in range(self.fc_env_lay_num):
                    layername = f"{self.name}_envfc{i}"
                    fc_env = layers.Dense(self.fc_env_sizes[i],activation = self.fc_env_actv[i],
                                          kernel_regularizer=regularizers.L2(self.fc_env_regu[i]),
                                          name = layername)(fc_env)
                env_dim_out = self.fc_env_sizes[-1]
            else:
                fc_env = input_env
                env_dim_out = self.env_dim

            env_reshape = layers.Reshape((env_dim_out,1),
                                    name = self.input_names[1]+"_"+self.name+"_rshpe")(fc_env)
        else:
            env_dim_out = 0
            
        # Conv layers:
        conv = imp_reshape
        dim_now = self.input_dim #dim_now record dimension of the current conv layer
        for i in range(self.conv_lay_num):
            
            if self.env_dim>0 and (i==0 or self.env_on_all_conv): 
                # Every input need to be blend with environment nodes before feeding into CNN layer

                env_for_conv = tf.repeat(env_reshape, 
                                         repeats=[self.conv_channels[i-1] if i>0 else 1], 
                                         axis=-1) # (env_dim,conv_channels[i-1])
                # env_for_conv = layers.Reshape((self.env_dim,self.conv_channels[i-1] if i>0 else 1))(env_for_conv)
                    ## (env_dim,conv_channels[i-1])
                comb_for_conv = layers.Concatenate(axis = -2,
                                                   name = f"{self.name}_conv{i}_in_concat")([conv,env_for_conv])
                                                 # (input_dim + env_dim, conv_channels[i-1])
                # Create gathering indexes and gather:
                idxes = []
                ## INDEX CANNNOT BE NEGATIVE
                idxes_env = [dim_now+j for j in range(env_dim_out)] #indexes where the evironments locate in concat tensor
                for j in range(dim_now):
                    idxes.append(j)
                    # Append indexes of evironments every conv_cluster[i] times
                    if (j+1)%self.conv_clusters[i] == 0:
                        idxes += idxes_env
                inp_for_conv = tf.gather(comb_for_conv,indices = idxes,axis = -2)
                conv = inp_for_conv
                
            # Build CNN        
            layername = f"{self.name}_conv{i}"
            conv = layers.Conv1D(self.conv_channels[i],
                                 self.conv_clusters[i] + (env_dim_out if (i==0 or self.env_on_all_conv) else 0),
                                 strides = self.conv_clusters[i] + (env_dim_out if (i==0 or self.env_on_all_conv) else 0),
                                 activation = self.conv_actv[i],
                                 padding = "valid",
                                 kernel_regularizer=regularizers.L2(self.conv_regu[i]),
                                 name = layername)(conv)
            dim_now = dim_now//self.conv_clusters[i] # Update dim_now 
            
        fla = layers.Flatten(name = f"{self.name}_fla")(conv)
        
        # FC layers:
        fc = fla
        for i in range(self.fc_lay_num):
            layername = f"{self.name}_fc{i}"
            fc = layers.Dense(self.fc_sizes[i],activation = self.fc_actv[i],
                              kernel_regularizer=regularizers.L2(self.fc_regu[i]),
                              name = layername)(fc)
            
        output_ptrn = layers.Dense(self.output_dim,activation = self.out_actv,
                                   name = self.output_name)(fc)
        
        # If there is prelayer and prelayer is not input layer, then only return output pattern layer
        if (prelayer != None) and ("in" not in self.input_names[0]):
                return (None,output_ptrn)
        else:
            if self.env_dim == 0:
                model = Model(inputs = input_imp, outputs = output_ptrn)
            else:
                model = Model(inputs = [input_imp,input_env], outputs = output_ptrn)
            return (model,output_ptrn)
        
    def load_net(self,emc_net_model,pre_net_dir,
                 lock_pre_weights = True,verbose = True):
        
        emc_net_model.load_weights(pre_net_dir,by_name = True)
        if verbose:
            print(f"prediction network {self.name} weights loaded.")
        if (not lock_pre_weights): return
        layer_names  = []
        for i in range(self.conv_lay_num):
            layername = f"{self.name}_conv{i}"
            layer_names.append(layername)
            lyhandle = emc_net_model.get_layer(name=layername)
            lyhandle.trainable = False
        for i in range(self.fc_lay_num):
            layername = f"{self.name}_fc{i}"
            layer_names.append(layername)
            lyhandle = emc_net_model.get_layer(name=layername)
            lyhandle.trainable = False
        layername = self.output_name
        layer_names.append(layername)
        lyhandle = emc_net_model.get_layer(name=layername)
        lyhandle.trainable = False
        if verbose:
            print(f"prediction network {self.name} weights locked:{layer_names}")