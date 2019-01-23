import numpy as np

class WaveFunction:
    def __init__(self, N, D, w):
        '''Constructor'''
        self.N = N
        self.D = D
        self.w = w
        Elements = ["Gauss", "PadeJastrow"]
        self.Elements = self.ExtractElements(Elements)
        
    def __call__(self, a, b, c, r, R):
        '''Calculate total wave function'''
        TotalWF = 1
        for obj in self.Elements:
            obj = eval(obj)
            TotalWF *= obj.WF(a, b, c, r, R)      
        return TotalWF*TotalWF
        
    def KineticEnergy(self, a, b, c, r, R):
        '''Calculate local kinetic energy'''
        TotalEnergy = 0
        for k in range(self.N):
            Energy_k = 0
            for obj in self.Elements:
                obj = eval(obj)
                Energy_k += obj.FirstDer(a, b, c, r, R, k)
            TotalEnergy += Energy_k * Energy_k
        for obj in self.Elements:
            obj = eval(obj)
            TotalEnergy += obj.SecondDer(a, b, c, r, R)
        return TotalEnergy
        
    def Gradient(self, a, b, c, r, R):
        '''Calculate derivatives used in optimization'''
        Energy = 0
        for obj in self.Elements:
            obj = eval(obj)
            for k in range(self.N):
                Energy += obj.FirstDer(a, b, c, r, R, k)
        gradients = []
        for obj in self.Elements:
            obj = eval(obj)
            gradients.append(obj.NablaSecond(a, b, c, r, R) + 2*Energy*obj.NablaFirst(a, b, c, r, R))
        return np.array(gradients)
        
    def ExtractElements(self, Elements):
        '''Transform list of elements to list of functions'''
        Objects = []
        for i in range(len(Elements)):
            if Elements[i] == "Gauss":
                Objects.append("Gauss(self.N, self.D, self.w)")
            elif Elements[i] == "PadeJastrow":
                Objects.append("PadeJastrow(self.N, self.D, self.w)")
            elif Elements[i] == "HydrogenLike":
                Objects.append("HydrogenLike(self.N, self.D, self.w)")
            else:
                Objects.append(Elements[i])
        return Objects
        

class Gauss(WaveFunction):
    def __init__(self, N, D, w):
        '''Constructor'''
        WaveFunction.__init__(self, N, D, w)

    def WF(self, a, b, c, r, R):
        '''Gaussian function'''
        return np.exp(-0.5 * a * np.sum(np.square(r)))
        
    def FirstDer(self, a, b, c, r, R, k):
        '''∇ln(WF) with respect to r_k'''
        return -a * r[k]
        
    def SecondDer(self, a, b, c, r, R):
        '''∇²ln(WF), sum over all r_k's'''
        return -a * self.N * self.D
        
    def NablaFirst(self, a, b, c, r, R):
        '''Derivative of ∇ln(WF) with respect to a'''
        return 0.5 * r.sum()
        
    def NablaSecond(self, a, b, c, r, R):
        '''Derivative of ∇²ln(WF) with respect to a'''
        return 0.5 * self.N * self.D 
        
        
class PadeJastrow(WaveFunction):
    def __init__(self, N, D, w):
        '''Constructor'''
        WaveFunction.__init__(self, N, D, w)
        
    def WF(self, a, b, c, r, R):
        '''Pade-Jastrow factor'''
        counter = 0
        for i in range(self.N):
            for j in range(i):
                counter += R[i,j]/(1 + b * R[i,j])
        return counter
                
    def FirstDer(self, a, b, c, r, R, k):
        '''∇ln(WF) with respect to r_k'''
        counter = 0
        for j in range(k):
            counter += 1/(1 + b * R[k,j])**2
        return counter
        
    def SecondDer(self, a, b, c, r, R):
        '''∇²ln(WF), sum over all r_k's'''
        counter = 0
        for k in range(self.N):
            for j in range(k):
                counter -= 2 * b/(1 + b * R[k,j])**3
        return counter
        
    def NablaFirst(self, a, b, c, r, R):
        '''Derivative of ∇ln(WF) with respect to b'''
        counter = 0
        for k in range(self.N):
            for j in range(k):
                counter += R[k,j]/(1 + b * R[k,j])**3
        return counter
        
    def NablaSecond(self, a, b, c, r, R):
        '''Derivative of ∇²ln(WF) with respect to b'''
        counter = 0
        for k in range(self.N):
            for j in range(k):
                counter += (R[k,j]/(1+b*R[k,j])**3)*((1-2*b*R[k,j])/(1+b*R[k,j]))
        return counter
        
        
class HydrogenLike(WaveFunction):
    def __init__(self, N, D, w):
        '''Constructor'''
        WaveFunction.__init__(self, N, D, w)
        
    def WF(self, a, b, c, r, R):
        '''Hydrogen-like orbitals'''
        return np.exp(-c * self.N * r.sum())
        
    def FirstDer(self, a, b, c, r, R, k):
        '''∇ln(WF) with respect to r_k'''
        return -c * self.N
        
    def SecondDer(self, a, b, c, r, R):
        '''∇²ln(WF), sum over all r_k's'''
        return 0
        
    def NablaFirst(self, a, b, c, r, R):
        '''Derivative of ∇ln(WF) with respect to b'''
        return 0.5 * self.N * self.N
        
    def NablaSecond(self, a, b, c, r, R):
        '''Derivative of ∇²ln(WF) with respect to b'''
        return 0
