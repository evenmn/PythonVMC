import numpy as np
from Optimization import *
from Metropolis import *

class VMC:
    def __init__(self, N, D, MC, MaxIter, w, dx, eta, tol):
        '''Constructor'''
        self.N          = N
        self.D          = D
        self.MC         = MC
        self.MaxIter    = MaxIter
        self.w          = w
        self.dx         = dx
        self.eta        = eta
        self.tol        = tol
        
    def SetSystem(self, Interaction=True, Potential="HarmonicOscillator", Sampling="BruteForce", \
                  Optimizer="GradientDescent"):
        '''Set System'''
        self.Interaction = Interaction
        self.Potential   = Potential
        self.Sampling    = Sampling
        self.Optimizer   = Optimizer
        
    def SetVariables(self, a=1.0, b=1.0, c=1.0, x='normal'):
        '''Set variables'''
        # Initialize positions
        if x == 'uniform':
            self.x = np.random.rand(self.N, self.D) - 0.5
        elif x == 'normal':
            self.x = np.random.normal(size=(self.N, self.D))
        self.R = np.zeros((self.N, self.N))                # Initialize rij
        self.r = np.zeros(self.N)                          # Initialize ri
        self.a = a                               # Initialize gaussian parameter
        self.b = b                               # Initialize Pade-Jastrow parameter
        self.c = c
        # Declare objects
        from LocalEnergy import LocalEnergy
        self.EL  = LocalEnergy(self.N, self.D, self.w, self.Potential, self.Interaction)
        self.Met = Metropolis(self.N, self.D, self.w, self.dx, self.Sampling)
        self.Opt  = Optimization(self.N, self.D, self.MC, self.w, self.eta, self.Optimizer)
        self.WF = WaveFunction(self.N, self.D, self.w)
        
    def Iterator(self):
        ''' Parameter update '''
        self.Energies = []
        for iter in range(self.MaxIter):
            E_tot   = 0           # Counter for total energy
            E_sqrd  = 0           # Counter for total energy squared
            grad_tot  = 0         # Counter for derivative of a
            gradE_tot = 0         # Counter for derivative of a multiplied with E
            for i in range(self.MC):
                nRand = np.random.randint(self.N)                   # Next particle to move
                dRand = np.random.randint(self.D)                   # Direction to move in
                xNew, rNew, RNew, PsiRatio = self.Met(self.x, self.r, self.R, self.a, self.b, self.c, nRand, dRand)      # Metropolis
                if(PsiRatio >= np.random.rand(1)):
                    self.x = xNew
                    self.r = rNew
                    self.R = RNew
                E  = self.EL(self.a, self.b, self.c, self.r, self.R)
                grad = self.WF.Gradient(self.a, self.b, self.c, self.r, self.R)
                E_tot   += E
                E_sqrd  += E*E
                grad_tot  += grad
                gradE_tot += grad*E
                
            EL_avg = E_tot/self.MC
            self.Energies.append(EL_avg)
            σ = np.sqrt(E_sqrd/self.MC - EL_avg*EL_avg)
            
            print("--- Iteration {} ---".format(iter+1))
            print("<E>: ", EL_avg)
            print("<σ>: ", σ, "\n")
            
            if iter > 1 and abs(EL_avg - self.Energies[-2]) < self.tol:
                print("=== Converged after {} iterations ===".format(iter+1))
                print("Final energy:   ", EL_avg)
                print("Final variance: ", σ)
                break
            
            dE = self.Opt(EL_avg, grad_tot, gradE_tot)      # Optimization
            self.a -= dE[0]
            self.b -= dE[1]
            #self.c -= dE[2]
        
    def Plotter(self):
        '''Plot energy'''
        import matplotlib.pyplot as plt
        plt.plot(self.Energies)
        plt.xlabel("Timestep")
        plt.ylabel("Energy [a.u.]")
        plt.grid()
        plt.show()
