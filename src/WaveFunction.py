import numpy as np

class WaveFunction:
    def __init__(self, N, D, w):
        '''Constructor'''
        self.N = N
        self.D = D
        self.w = w
        
    def __call__(self, a, b, r, R):
        '''Calculate total wave function'''
        # Specify wave function elements
        objects = [Gauss(self.N, self.D, self.w)]
        TotalWF = 1
        for obj in objects:
            TotalWF *= obj.WF(a, r)      
        return TotalWF*TotalWF
        
    def KineticEnergy(self, a, b, r, R):
        '''Calculate local kinetic energy'''
        # Specify wave function elements
        objects = [Gauss(self.N, self.D, self.w)]
        TotalEnergy = 0
        for k in range(self.N):
            Energy_k = 0
            for obj in objects:
                Energy_k += obj.FirstDer(a, r, k)
            TotalEnergy += Energy_k * Energy_k
        for obj in objects:
            TotalEnergy += obj.SecondDer(a, r)
        return TotalEnergy
        
    def Gradient(self, a, r):
        '''Calculate derivatives used in optimization'''
        # Specify wave function elements
        objects = [Gauss(self.N, self.D, self.w)]
        
        gradients = []
        for obj in objects:
            gradients.append(obj.Nabla(a, r))
        return np.array(gradients)[0]
        

class Gauss(WaveFunction):
    def __init__(self, N, D, w):
        '''Constructor'''
        WaveFunction.__init__(self, N, D, w)

    def WF(self, a, r):
        '''Gaussian function'''
        return np.exp(-0.5 * a * np.sum(np.square(r)))
        
    def FirstDer(self, a, r, k):
        '''First derivative of ln(WF) with respect to r_k'''
        return -a * r[k]
        
    def SecondDer(self, a, r):
        '''Second derivative og ln(WF), sum over all r_k's'''
        return -a * self.N * self.D
        
    def Nabla(self, a, r):
        '''Derivative of energy with respect to a'''
        return 3*self.N - 4*a*a*np.sum(np.square(r))
        
        
class PadeJastrow(WaveFunction):
    def __init__(self, N, D, w):
        '''Constructor'''
        WaveFunction.__init__(self, N, D, w)
        
    def WF(self, b, R):
        '''Pade-Jastrow factor'''
        for i in range(self.N):
            for j in range(i):
                return R(i,j)/(1 + b * R(i,j))
                
    def FirstDer(self, a, r, k):
        '''First derivative of ln(WF) with respect to r_k'''
        return 1
        
    def SecondDer(self, a, r, k):
        '''Second derivative og ln(WF), sum over all r_k's'''
        return 1
