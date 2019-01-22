from MonteCarlo import *

FermionsInHO = VMC(N        = 2, 
                   D        = 2, 
                   MC       = 10000, 
                   MaxIter  = 100, 
                   w        = 1, 
                   dx       = 0.1, 
                   eta      = 0.01,
                   tol      = 1e-6)

FermionsInHO.SetSystem(Interaction = False, 
                       Potential   = "HarmonicOscillator", 
                       Sampling    = "BruteForce",
                       Optimizer   = "GradientDescent")

FermionsInHO.SetVariables(a=3)
FermionsInHO.Iterator()
FermionsInHO.Plotter()
