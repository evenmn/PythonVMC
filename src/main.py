from MonteCarlo import *

FermionsInHO = VMC(N        = 2, 
                   D        = 2, 
                   MC       = 10000, 
                   MaxIter  = 30, 
                   w        = 1, 
                   dx       = 0.1, 
                   eta      = 0.05,
                   tol      = 1e-4)

FermionsInHO.SetSystem(Interaction = True, 
                       Potential   = "HarmonicOscillator", 
                       Sampling    = "BruteForce",
                       Optimizer   = "GradientDescent",
                       Elements    = ["Gauss", "PadeJastrow"])

FermionsInHO.SetVariables(a=2)
FermionsInHO.Iterator()
FermionsInHO.Plotter()
