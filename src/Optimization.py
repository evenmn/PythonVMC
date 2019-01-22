import numpy as np

class Optimization:
    def __init__(self, N, D, MC, w, eta, Optimizer):
        '''Constructor'''
        self.N   = N
        self.D   = D
        self.MC  = MC
        self.w   = w
        self.eta = eta
        self.Optimizer = Optimizer
        
    def __call__(self, EL_avg, da_tot, daE_tot):
        '''Returns correlation term'''
        if self.Optimizer == "GradientDescent":
            return self.GradientDescent(EL_avg, da_tot, daE_tot)
        elif self.Optimizer == "ADAM":
            raise NotImplementedError("ADAM has yet to be implemented")

    def GradientDescent(self, EL_avg, da_tot, daE_tot):
        '''Gradient descent'''
        return 2*self.eta*(daE_tot - EL_avg*da_tot)/self.MC;
        
    
        
    
