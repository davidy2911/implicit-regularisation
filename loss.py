import numpy as np
import scipy as sp
import sympy as sym

a, b, x, y, u, v = sym.symbols('a b x y u v')

class loss:
    def __init__(self, E, data=[x,y]):
        self.formula = E.subs([(x, data[0]), (y, data[1])])

        self.gradient = np.array([[sym.diff(self.formula, a)], [sym.diff(self.formula, b)]])

        self.global_minima = sym.solve(self.formula, b)[0]

    def value(self, params):
        return self.formula.subs([(a, params[0]),(b, params[1])])
    
    def plot(self, range_a, range_b, include_minima=False):
        p_1 = sym.plotting.plot3d(self.formula, (a, range_a[0],range_a[1]), (b, range_b[0], range_b[1]), show =False)
        if include_minima == True:
            p_2 = sym.plotting.plot3d_parametric_line((u,self.global_minima.subs(a,u), 0.0000001*u, (u, 0.2, 5)), (v,self.global_minima.subs(a,v), 0.0000001*v, (v, -5, -0.2)), show =False)  
            p_2[0].line_color='r'   
            p_2[1].line_color='r' 
            p_1.extend(p_2)

        p_1.show()



def loss_modifier(og_loss, h):
    regulizer = (h/4)*(og_loss.gradient[0][0]**2 + og_loss.gradient[1][0]**2)
    modified_formula = og_loss.formula + regulizer

    return loss(modified_formula)
        
if __name__ == '__main__':
    E = (y-a*b*x)**2/2
    print(E)
    
    data = np.array([1, 0.6])
    print(data)

    og_loss = loss(E, data)

    print(og_loss.formula)
    print(og_loss.gradient)
    
    modified_loss = loss_modifier(og_loss, 0.025)

    print(modified_loss.formula)
    print(modified_loss.gradient)
    
    params = np.array([2.8, 3.5])
    print(og_loss.value(params))
    print(modified_loss.value(params))
    
    og_loss.plot((-5,5),(-5,5))
    modified_loss.plot((-5,5),(-5,5))

    print(og_loss.global_minima)
