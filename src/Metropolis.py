import numpy as np
from WaveFunction import *

class Metropolis:
    def __init__(self, N, D, w, dx, Sampling, Elements):
        '''Constructor'''
        self.N = N
        self.D = D
        self.w = w
        self.dx = dx 
        self.Sampling = Sampling
        self.Psi = WaveFunction(N, D, w, Elements)
        
    def __call__(self, x, r, R, a, b, c, nRand, dRand):
        '''Updates position and PsiRatio'''

        if self.Sampling == "BruteForce":
            return self.BruteForce(x, r, R, a, b, c, nRand, dRand)
        elif self.Sampling == "ImportanceSampling":
            return self.ImportanceSampling(x, r, R, a, b, c, nRand, dRand)

    def BruteForce(self, x, r, R, a, b, c, nRand, dRand):
        '''Brute Force Metropolis algorithm'''
        xNew = x.copy()
        xNew[nRand, dRand] = x[nRand, dRand] + (np.random.rand(1)-0.5) * self.dx
        rNew = self.Dist(xNew)
        RNew = self.Diff(xNew)
    
        PsiRatio = self.Psi(a, b, c, rNew, RNew)/self.Psi(a, b, c, r, R)
        return xNew, rNew, RNew, PsiRatio
        
    def ImportanceSampling(self, x, r, R, a, b, c, nRand, dRand):
        '''Importance Sampling algorithm'''
        xNew = x.copy()
        rNew = self.Dist(xNew)
        RNew = self.Diff(xNew)
        
        PsiRatio = self.Psi(a, b, c, rNew, RNew)/self.Psi(a, b, c, r, R)
        return xNew, rNew, RNew, PsiRatio
        
    def Diff(self, x):
        '''Calculate distance matrix between particles'''
        R = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i):
                count = 0
                for d in range(self.D):
                    count += (x[i,d] - x[j,d])**2
                R[i,j] = np.sqrt(count)
        return R
        
    def Dist(self, x):
        '''Calculate distance from origin'''
        r = np.zeros(self.N)
        for i in range(self.N):
            DistPerParticle = 0
            for d in range(self.D):
                DistPerParticle += x[i,d]**2
            r[i] = DistPerParticle
        return r
