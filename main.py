import numpy as np
import scipy as sp
import sys
import sympy as sym

import loss as ls 

a, b, x, y = sym.symbols('a b x y')

E = ((y-a*b*x)**2)/2
print("\n Formula for loss space for any data (x,y)")
print (E)

data = np.array([1,0.6])
print("\nGiven data:")
print(data)

original_loss = ls.loss(E, data)

print("\nFormula for original loss space for given data:")
print(original_loss.formula)

modified_loss = ls.loss_modifier(original_loss, 0.25)

print("\nFormula for modified version of original loss space:")
print(modified_loss.formula)

range_a =(-5,5)
range_b =(-5,5)

print("\nPlot of original loss space over given range:\n")
original_loss.plot(range_a, range_b, include_minima = True)
print("\nPlot of modified loss space over given range:\n")
modified_loss.plot(range_a, range_b, include_minima = True)
