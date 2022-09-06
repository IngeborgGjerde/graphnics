from fenics import *
from xii import *
import sys
sys.path.append('../') 
from graphnics import *

def time_stepping_stokes(model, rho, t_steps, T, qp_n=None):
    '''
    Do time stepping for models of the type
        rho/A d/dt q + a(U, V) = L(V;f,g,ns)
    on graph G, where U=(q, ..., p) and V=(v, ..., phi)
    
    Args:
        model (class): class containing G, W, a and L
        t_steps (int): number of time steps
        T (float): end time
    '''
    
    if qp_n is None: qp_n = ii_Function(model.W) # initialize as zero
    
    
    # split out the components
    qs = model.qp[0:model.G.num_edges]
    vs = model.vphi[0:model.G.num_edges]

    dt = Constant(T/t_steps)
    dt_val = dt(0)
    
    
    # We discretize the time derivative term as rho/A d/dt q = rho/A (qn1 - qn)/Delta t
    aa = model.a_form()
    LL = model.L_form()

    for i, e in enumerate(model.G.edges):

        dx_edge = Measure("dx", domain = model.G.edges[e]['submesh'])
        Ainv = Constant(1)# TODO: change to model.G.edges[e]['Ainv']

        aa[i][i] += rho*Ainv*qs[i]*vs[i]*dx_edge # qn1 belongs to the lhs as it is unknown
        LL[i] += rho*Ainv*qp_n[i]*vs[i]*dx_edge # qn belongs to the rhs as it is known

    # Finally we time step
    qps = []
    
    # Update f and n to next time step
    model.f.t, model.p_bc.t = 0, 0
    b = ii_convert(ii_assemble(LL))

    model.f.t, model.p_bc.t = dt_val, dt_val
    A = ii_convert(ii_assemble(aa))

    for t in np.linspace(dt_val, T, t_steps-1):

        sol = ii_Function(model.W)
        solver = LUSolver(A, 'mumps')
        solver.solve(sol.vector(), b)

        qps.append(sol) 

        # Update qp_n
        for i, func in enumerate(sol):
            qp_n[i].assign(func)

        model.f.t, model.p_bc.t = t, t

        b = ii_convert(ii_assemble(LL))
        
        # Update f and n to next time step
        model.f.t, model.p_bc.t = t+dt_val, t+dt_val
    
        A = ii_convert(ii_assemble(aa))

    return qps
