import networkx as nx
from fenics import *
import sys
sys.path.append('../../')

from graphnics import *
from networkmodels import *
import numpy as np

''''
Simulate vasomotion in honeycomb pia

Args:
    t_step_scheme (str): either 'CN' (Crank-Nicholson) or 'IE' (implicit Euler)
    t_steps (int): number of time steps
    bifurcations (int): number of bifurcations
'''
    
        
def resistance(radius1, mu):
    '''
    Args:
        radius1 (float): inner radius
        mu (float): fluid viscosity
    '''
    Q = 15.5 # calculated from pial artery cross-section (geometry from Vinje et. al.)
    return mu/(Q*radius1**4)

def area(radius1, radius2):
    return np.pi*radius2**2-np.pi*radius1**2
    
if __name__ == '__main__':
    
    fluid_params = {}
    fluid_params["rho"] = 1e-3
    fluid_params["nu"] = 0.697
    mu = fluid_params["nu"]*fluid_params["rho"]
    fluid_params["mu"] = mu

    t_steps = 4
    
    n_combs = [1, 2, 3]
    
    for n_comb in n_combs:
        G = honeycomb(n_comb, n_comb)
        G.make_mesh(4)
        
        print(f'Network has {G.num_edges:g} edges')
        
        p_static_gradient = 0#0.1995
        p_outlet = -n_comb*p_static_gradient
        p_bc = Expression('x[0]*deltap', degree=1, deltap = p_outlet)

        # Assign inner and outer radius to each edge
        radius_val = 0.05 # 50 micrometers
        prop_dict = {key: 
                            {
                            'radius1':Constant(radius_val), 
                            'radius2':Constant(radius_val*3),
                            'res':Constant(resistance(radius_val, mu)),
                            'Ainv':Constant(1/area(radius_val, radius_val*3))
                            } 
                    for key in list(G.edges.keys())}
        
        nx.set_edge_attributes(G, prop_dict)
        
        
        # We scale our mesh to be roughly 1 mm per honeycomb 
        
        f = Expression('sin(t)', degree=2, t=0)
        
        model = NetworkStokes(G, fluid_params, f, p_bc)
        q0 = [Constant(1e-6)]
        
        
        qps = time_dep_stokes(G, model, t_steps=t_steps, T=6.28, q0=q0, p0=p_bc, t_step_scheme='IE')


        file_q = File('plots/q.pvd')
        file_p = File('plots/p.pvd')
        
        for i, qp in enumerate(qps):
            
            qs = qp.split()[0:G.num_edges]
            p = qp.split()[-1]
            q = GlobalFlux(G, qs)
            qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'CG', 2, G.geom_dim))
            
            qi.rename('q', '0.0')
            p.rename('p', '0.0')
            
            file_q << (qi, float(i))
            file_p << (p, float(i))
        
