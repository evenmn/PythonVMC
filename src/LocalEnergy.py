import numpy as np
from MonteCarlo import VMC
from WaveFunction import *

class LocalEnergy(VMC):
    def __init__(self, N, D, w, Potential, Interaction, Elements):
        '''Constructor'''
        self.N = N
        self.D = D
        self.w = w
        self.Potential = Potential
        self.Interaction = Interaction
        self.Elements = Elements
        
    def __call__(self, a, b, r, R):
        '''Total local energy'''
        Pot = Potential(self.N, self.D, self.w, self.Potential, self.Interaction, self.Elements)
        Int = Interaction(self.N, self.D, self.w, self.Potential, self.Interaction, self.Elements)
        Kin = Kinetic(self.N, self.D, self.w, self.Potential, self.Interaction, self.Elements)
        return Kin(a, b, r, R) + Pot(r) + Int(R)
    
    
class Kinetic(LocalEnergy):
    def __init__(self, N, D, w, Potential, Interaction, Elements):
        '''Constructor'''
        LocalEnergy.__init__(self, N, D, w, Potential, Interaction, Elements)
        
    def __call__(self, a, b, r, R):
        '''Total kinetic energy of system'''
        system = 0
        if system == 0:
            return self.Laplacian(a, b, r, R)
            
    def Laplacian(self, a, b, r, R):
        '''Laplace operator'''
        WF = WaveFunction(self.N, self.D, self.w, self.Elements)
        return -0.5 * WF.KineticEnergy(a, b, r, R, )
        
        
class Potential(LocalEnergy):
    def __init__(self, N, D, w, Potential, Interaction, Elements):
        '''Constructor'''
        LocalEnergy.__init__(self, N, D, w, Potential, Interaction, Elements)
        
    def __call__(self, r):
        '''Total external potential of system'''
        if self.Potential == "HarmonicOscillator":
            return self.HarmonicOscillator(r)  
        elif self.Potential == "AtomicNucleus":
            return self.AtomicNucleus(r)

    def HarmonicOscillator(self, r):
        '''Harmonic oscillator potential'''
        return 0.5 * self.w * self.w * np.sum(np.square(r))
        
    def AtomicNucleus(self, r):
        '''Atomic potential'''
        return 0.5 * self.w * self.w * np.reciprocal(R).sum()
    
    
class Interaction(LocalEnergy):
    def __init__(self, N, D, w, Potential, Interaction, Elements):
        '''Constructor'''
        LocalEnergy.__init__(self, N, D, w, Potential, Interaction, Elements)
        
    def __call__(self, R):
        '''Total interaction energy of system'''
        if self.Interaction:
            return self.Coulomb(R)
        else:
            return 0
            
    def Coulomb(self, R):
        '''Interaction energy'''
        counter = 0
        for i in range(self.N):
            for j in range(i):
                counter += 1/R[i,j]
        return counter
