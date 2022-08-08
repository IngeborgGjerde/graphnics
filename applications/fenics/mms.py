from fenics import *
import sympy as sym
import numpy as np

def coupled_1D_3D(a, b, R):
    '''
    Get manufactured solution for 1 line going from (0,0,a) to (0,0,b)
    '''
    x, y, z = sym.symbols('x[0] x[1] x[2]')

    # Analytic solutions
    r2 =  (x-a[0])**2.0 + (y-a[1])**2.0 + DOLFIN_EPS
    r_a, r_b = [sym.sqrt( r2 + (z-xx[2])**2.0) for xx in [a, b]]

    G = 1.0/(4.0*np.pi)*(sym.ln( r_b - (z-b[2]) ) - sym.ln(r_a - (z-a[2])))
    
    u_a = G
    
    # u_a = G (which satisfies -Delta u = \delta_\Lambda) implies beta(hat_u-bar_u)=1
    # meaning that beta=1/(hat_u-bar_u)
    
    R_a, R_b = [sym.sqrt( R**2.0 + (z-xx[2])**2.0 + DOLFIN_EPS) for xx in [a,b] ]
    G_R = 1.0/(4.0*np.pi)*(sym.ln( R_b - (z-b[2])) - sym.ln(R_a - (z-a[2])))
    u_bar_a = G_R
    
    # We make u_hat_a that satisfies -Delta hat_u = -1
    u_hat_a = 1 + 0.5*z**2
    
    beta = 1/(u_hat_a - u_bar_a) # Generate beta so that u satisfies the original equation

    u_a, u_hat_a, beta = [Expression(sym.printing.ccode(func).replace('log', 'std::log'), degree=3) for func in [u_a, u_hat_a, beta]]
    
    return u_a, u_hat_a, beta



def time_dep_line_source(a, b):
    '''
    Get manufactured solution time dependent Poisson equation with line source
    '''
    x, y, z, t = sym.symbols('x[0] x[1] x[2] t')

    r2 =  (x-a[0])**2.0 + (y-a[1])**2.0 + DOLFIN_EPS
    r_a, r_b = [sym.sqrt( r2 + (z-xx[2])**2.0) for xx in [a, b]]

    G = 1.0/(4.0*np.pi)*(sym.ln( r_b - (z-b[2]) ) - sym.ln(r_a - (z-a[2])))

    f_line = 100*sym.sin(t)
    
    u_a = f_line*G
    
    f3 = sym.diff(u_a, t)
    
    u_a, f_line, f3 = [Expression(sym.printing.ccode(func).replace('log', 'std::log'), degree=3, t=0) for func in [u_a, f_line, f3]]
    
    return u_a, f_line, f3
