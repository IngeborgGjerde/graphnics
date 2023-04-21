'''
Copyright (C) 2022-2023 by Ingeborg Gjerde

This file is a part of the graphnics project (https://arxiv.org/abs/2212.02916)

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.
'''


from fenics import *
import sys
sys.path.append("../")
set_log_level(40)
from graphnics import *
import ufl

def hydraulic_manufactured_solution(G, Ainv, Res):
    '''
    Make manufactured solution for hydraulic network model
        1/A \partial_t q + Rq + \nabla p = g
        \nabla\cdot q = f
        
    Args:
        G (networkx graph): graph to solve on
        Ainv (float): inverse of cross sectional area
        Res (float): resistance
        
    Returns:
        f, q, p, g: ufl functions for the manufactured solution
        t: time variable
    '''

    # some nice solution
    t = Constant(0)
    
    xx = SpatialCoordinate(G.mesh)
    
    q = t*sin(2*3.14*xx[0])# + cos(2*3.14*t)
    p = t*cos(2*3.14*xx[0])# + sin(2*3.14*t)
    
    g = Ainv*ufl.diff(q,t) + Res*q + G.dds(p)
    f = G.dds(q)
    
    return q, p, f, g, t


def sympy_hydraulic_manufactured_solution(G, Ainv, Res):
    
    import sympy as sym
    x, t = sym.symbols('x[0] t')
 
    q = t*sym.sin(2*3.14*x)
    p = sym.cos(2*3.14*x)
    
    g = Ainv*sym.diff(q,t) + Res*q + sym.diff(p,x)
    f = sym.diff(q,x)
    
    q, p, f, g = [Expression(sym.printing.ccode(expr), degree=2, t=0) for expr in (q, p, f, g)]
    
    return q, p, f, g   


def test_time_stepping_hydraulic():
    
    G = line_graph(2)
    G.make_mesh(10)
    
    Ainv = 3
    Res = 10
    
    T = 1
    t_steps = 100
    
    G.make_mesh(8)
    
    # Make solution of time dependent stokes
    q, p, f, g, t = hydraulic_manufactured_solution(G, Ainv, Res)    
    
    for time_step_scheme in ['IE', 'CN']:
        t.assign(0) # set time to 0
        model = TimeDepHydraulicNetwork(G, f=f, g=g, p_bc=p, Res = Res, Ainv=Ainv)

        qps = time_stepping_stokes(model, t, t_steps=t_steps, T=T, t_step_scheme=time_step_scheme)
        qh, ph = qps[-1] # last time step solution

        t.assign(T) # set time to T
        
        qa = project(q, FunctionSpace(G.mesh, "CG", 3))
        pa = project(p, FunctionSpace(G.mesh, "CG", 2))
    
        error_q = errornorm(qh, qa)
        error_p = errornorm(ph, pa)
        
        assert error_q < 1e-2, f'Large error in time stepping hydraulic model with {time_step_scheme},  error_q={error_q:1.1e}'
        assert error_p < 1.5e-2, f'Large error in time stepping hydraulic model with {time_step_scheme}, error_q={error_p:1.1e}'
    
    
def test_time_stepping_mixed_hydraulic():
    
    G = line_graph(2)
    G.make_mesh(5)
    G.make_submeshes()
    
    Ainv = 2
    Res = 10
    
    for e in G.edges():
        G.edges()[e]['Ainv'] = Ainv
        G.edges()[e]['Res'] = Res
        
    T = 1
    t_steps = 10
    
    q, p, f, g = sympy_hydraulic_manufactured_solution(G, Ainv, Res)
    
    t = Constant(0)
    
    t.assign(0) # set time to 0
    
    model = TimeDepMixedHydraulicNetwork(G, f=f, g=g, p_bc=p, Ainv=Ainv)

    # TODO: Test with CN
    qps = time_stepping_stokes(model, t, t_steps=t_steps, T=T, t_step_scheme="IE")
    qh, ph = qps[-1] # last time step solution

    q.t = T
    p.t = T
    
    qa = project(q, FunctionSpace(G.mesh, "CG", 3))
    pa = project(p, FunctionSpace(G.mesh, "CG", 2))
    
    error_q = errornorm(qh, qa)
    error_p = errornorm(ph, pa)
    
    assert error_q < 1e-1, f'Large error in time stepping mixed hydraulic model, error_q={error_q:1.1e}'
    assert error_p < 1e-1, f'Large error in time stepping mixed hydraulic model, error_q={error_p:1.1e}'
    
    #print(f'Dual, error_q={error_q:1.1e}, error_p={error_p:1.1e}')
    
    #import matplotlib.pyplot as plt
    #plt.plot(qh.function_space().tabulate_dof_coordinates()[:,0], qh.vector().get_local(), '.', label='qh')
    #plt.plot(qa.function_space().tabulate_dof_coordinates()[:,0], qa.vector().get_local(), label='qa')
    #plt.legend()
    #plt.savefig('test_time_stepping_flow.png')

    #plt.figure()
    #plt.plot(ph.function_space().tabulate_dof_coordinates()[:,0], ph.vector().get_local(), '.', label='ph')
    #plt.plot(pa.function_space().tabulate_dof_coordinates()[:,0], pa.vector().get_local(), label='pa')
    #plt.legend()
    #plt.savefig('test_time_stepping_pressure.png')
    
    
if __name__ == "__main__":
    pass
    #test_time_stepping_hydraulic()
    #test_time_stepping_mixed_hydraulic()