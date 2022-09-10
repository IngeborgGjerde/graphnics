from fenics import *
from xii import *
import sys
sys.path.append('../') 
from graphnics import *

time_stepping_schemes = {'IE':{'b1':0, 'b2':1},
                         'CN':{'b1':0.5, 'b2':0.5}}
    

# TODO: time stepping does not converge for CN?

def time_stepping_stokes(model, rho=Constant(1), t_steps=10, T=1, qp_n=None, t_step_scheme='IE'):
    '''
    Do time stepping for models of the type
        rho/A d/dt q + a(U, V) = L(V;f,g,ns)
    on graph G, where U=(q, ..., p) and V=(v, ..., phi)
    
    Args:
        model (class): class containing G, W, a and L
        rho (float): density
        t_steps (int): number of time steps
        T (float): end time
        qp_n (ii_function): initial solution
        t_step_scheme (str): 'IE' for Implicit Euler or 'CN' for Crank-Nicholson
    '''

    if qp_n is None: qp_n = ii_Function(model.W) # initialize as zero
    
    # split out the components
    qs = model.qp[0:model.G.num_edges]
    vs = model.vphi[0:model.G.num_edges]

    dt = T/t_steps
    
    # We discretize the time derivative term as rho/A d/dt q = rho/A (qn1 - qn)/Delta t
    cn, cn1 = time_stepping_schemes[t_step_scheme].values()
    
    Dn1 = [[ 0 for i in range(0, len(model.W))  ] for j in range(0, len(model.W))]
    Dn = [ 0 for i in range(0, len(model.W))  ]
    
    for i, e in enumerate(model.G.edges):
    
        dx_edge = Measure("dx", domain = model.G.edges[e]['submesh'])
        Ainv = Constant(1) # TODO: change to model.G.edges[e]['Ainv']
        
        Dn1[i][i] += rho*Ainv*qs[i]*vs[i]*dx_edge  
        Dn[i]     += rho*Ainv*qp_n[i]*vs[i]*dx_edge 


    for i in range(model.G.num_edges, len(model.W)):
        Dn1[i][i] += Constant(0)*model.qp[i]*model.vphi[i]*dx
        Dn[i]     += Constant(0)*model.vphi[i]*dx

    # Finally we time step
    qp_0 = ii_Function(model.W)
    for i, f0 in enumerate(qp_0):
        f0.vector()[:] = interpolate(qp_n[i], model.W[i]).vector().get_local()

    qps = [qp_0]

    a = model.a_form()
    L = model.L_form()

    # Update f and n to next time step
    model.f.t, model.p_bc.t = 0, 0
    An, Ln, DDn = [ii_convert(ii_assemble(term)) for term in [a, L, Dn]]
    
    model.f.t, model.p_bc.t = dt, dt
    An1, Ln1, DDn1 = [ii_convert(ii_assemble(term)) for term in [a, L, Dn1]]

    import numpy as np
    np.set_printoptions(3)

    for t in np.linspace(dt, T, t_steps-1):

        A = ii_convert( DDn1 + cn1*dt*An1 ) 
        b = ii_convert( DDn  + cn1*dt*Ln1 - dt*cn*An*qp_n.vector() + dt*cn*Ln )

        sol = ii_Function(model.W)  
        solver = LUSolver(A, 'mumps')
        solver.solve(sol.vector(), b)

        qps.append(sol) 
        # Update qp_n

        [qp_n[i].assign(func) for i, func in enumerate(sol)]
        
        # Update f and n to next time step
        model.f.t, model.p_bc.t = t,t
        An, Ln, DDn = [ii_convert(ii_assemble(term)) for term in [a, L, Dn]]
        
        model.f.t, model.p_bc.t = t+dt, t+dt
        An1, Ln1, DDn1 = [ii_convert(ii_assemble(term)) for term in [a, L, Dn1]]
        
    return qps
