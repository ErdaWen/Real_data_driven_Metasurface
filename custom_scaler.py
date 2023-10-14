import numpy as np

class Custom_Scaler:
    """ A minmax scaler with defined range
    
    Methodes
    ------------------------
        Transform: transform()
        Inverse transform: transform()
        
    """
    def __init__(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val
        
    def transform(self,matrix):
        matrix_new = np.copy(matrix)
        matrix_new = (matrix_new-self.min_val)/(self.max_val-self.min_val)
        return matrix_new
    
    def inverse_transform(self,matrix):
        matrix_new = np.copy(matrix)
        matrix_new = matrix_new*(self.max_val-self.min_val)+self.min_val
        return matrix_new