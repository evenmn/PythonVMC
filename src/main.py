'''
Simple VMC solver written in Python

Author: Even M. Nordhagen

Arguments:
---------

N:      {int}    Number of particles  
D:      {int}    Number of dimensions
MC:     {int}    Number of Monte-Carlo cycles
MaxIter {int}    Max number of iterations
w:      {float}  Harmonic oscillator frequency
dx:     {float}  Step length used when moving particle
eta:    {float}  Learning rate used in gradient optimization
tol:    {float}  Tolerance when deciding convergence
'''

from MonteCarlo import *

FermionsInHO = VMC(N        = 1, 
                   D        = 3, 
                   MC       = 100000, 
                   MaxIter  = 100, 
                   w        = 1, 
                   dx       = 0.1, 
                   eta      = 0.05,
                   tol      = 1e-4)

FermionsInHO.SetSystem(Interaction = False, 
                       Potential   = "AtomicNucleus", 
                       Sampling    = "BruteForce",
                       Optimizer   = "GradientDescent",
                       Elements    = ["HydrogenLike"]) #, "PadeJastrow"])

FermionsInHO.SetVariables(a=1, b=1, c=1)
FermionsInHO.Iterator()
FermionsInHO.Plotter()
