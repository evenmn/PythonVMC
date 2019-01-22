import numpy as np

class WaveFunction:
    def __init__(self, N, D, w, Elements):
        '''Constructor'''
        self.N = N
        self.D = D
        self.w = w
        self.Elements = self.ExtractElements(Elements)
        
    def __call__(self, a, b, r, R):
        '''Calculate total wave function'''
        TotalWF = 1
        for element in self.Elements:
            obj = eval(element)
            TotalWF *= obj.WF(a, b, r, R)      
        return TotalWF*TotalWF
        
    def KineticEnergy(self, a, b, r, R):
        '''Calculate local kinetic energy'''
        # Specify wave function elements
        objects = [Gauss(self.N, self.D, self.w, self.Elements),
                   PadeJastrow(self.N, self.D, self.w, self.Elements)]
        TotalEnergy = 0
        for k in range(self.N):
            Energy_k = 0
            for element in self.Elements:
                obj = eval(element)
                Energy_k += obj.FirstDer(a, b, r, R, k)
            TotalEnergy += Energy_k * Energy_k
        for element in self.Elements:
            obj = eval(element)
            TotalEnergy += obj.SecondDer(a, b, r, R)
        return TotalEnergy
        
    def Gradient(self, a, b, r, R):
        '''Calculate derivatives used in optimization'''
        # Specify wave function elements
        objects = [Gauss(self.N, self.D, self.w, self.Elements),
                   PadeJastrow(self.N, self.D, self.w, self.Elements)]
        Energy = 0
        for element in self.Elements:
            obj = eval(element)
            for k in range(self.N):
                Energy += obj.FirstDer(a, b, r, R, k)
        gradients = []
        for obj in objects:
            gradients.append(obj.NablaSecond(a, b, r, R) + 2*Energy*obj.NablaFirst(a, b, r, R))
        return np.array(gradients)
        
    def ExtractElements(self, Elements):
        ''' '''
        Objects = []
        for i in range(len(Elements)):
            if Elements[i] == "Gauss":
                Objects.append("Gauss(self.N, self.D, self.w, self.Elements)")
            elif Elements[i] == "PadeJastrow":
                Objects.append("PadeJastrow(self.N, self.D, self.w, self.Elements)")
            else:
                Objects.append(Elements[i])
        return Objects
        

class Gauss(WaveFunction):
    def __init__(self, N, D, w, Elements):
        '''Constructor'''
        WaveFunction.__init__(self, N, D, w, Elements)

    def WF(self, a, b, r, R):
        '''Gaussian function'''
        return np.exp(-0.5 * a * np.sum(np.square(r)))
        
    def FirstDer(self, a, b, r, R, k):
        '''∇ln(WF) with respect to r_k'''
        return -a * r[k]
        
    def SecondDer(self, a, b, r, R):
        '''∇²ln(WF), sum over all r_k's'''
        return -a * self.N * self.D
        
    def NablaFirst(self, a, b, r, R):
        '''Derivative of ∇ln(WF) with respect to a'''
        return 0.5 * r.sum()
        
    def NablaSecond(self, a, b, r, R):
        '''Derivative of ∇²ln(WF) with respect to a'''
        return 0.5 * self.N * self.D 
        
        
class PadeJastrow(WaveFunction):
    def __init__(self, N, D, w, Elements):
        '''Constructor'''
        WaveFunction.__init__(self, N, D, w, Elements)
        
    def WF(self, a, b, r, R):
        '''Pade-Jastrow factor'''
        counter = 0
        for i in range(self.N):
            for j in range(i):
                counter += R[i,j]/(1 + b * R[i,j])
        return counter
                
    def FirstDer(self, a, b, r, R, k):
        '''∇ln(WF) with respect to r_k'''
        counter = 0
        for j in range(k):
            counter += 1/(1 + b * R[k,j])**2
        return counter
        
    def SecondDer(self, a, b, r, R):
        '''∇²ln(WF), sum over all r_k's'''
        counter = 0
        for k in range(self.N):
            for j in range(k):
                counter -= 2 * b/(1 + b * R[k,j])**3
        return counter
        
    def NablaFirst(self, a, b, r, R):
        '''Derivative of ∇ln(WF) with respect to b'''
        counter = 0
        for k in range(self.N):
            for j in range(k):
                counter += R[k,j]/(1 + b * R[k,j])**3
        return counter
        
    def NablaSecond(self, a, b, r, R):
        '''Derivative of ∇²ln(WF) with respect to b'''
        counter = 0
        for k in range(self.N):
            for j in range(k):
                counter += (R[k,j]/(1+b*R[k,j])**3)*((1-2*b*R[k,j])/(1+b*R[k,j]))
        return counter
