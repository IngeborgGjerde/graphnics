import networkx as nx
from fenics import *
from fenics_graph import *
from utils import *
from graph_examples import *
    


def network_stokes_model(G, fluid_params, f=Constant(0), inlets=[], outlets=[]):

    mesh = G.global_mesh
    rho = Constant(fluid_params['rho'])
    nu = Constant(fluid_params['nu'])
    mu=nu*rho
    Res = Constant(1)

    Ainv = Constant(1)
    dt = 0.1
    T = 1


    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] 
    
    # Real space on each bifurcation, ordered by G.bifurcation_ixs
    LMs = [FunctionSpace(mesh, 'R', 0) for b in G.bifurcation_ixs] 

    # Pressure space on global mesh
    P1 = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh)
    
    ### Function spaces
    spaces = P2s + LMs + [P1]
    W = MixedFunctionSpace(*spaces) 


    # Trial and test functions
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)

    qp_n = Function(W)

    # split out the components
    qs = qp[0:G.num_edges]
    lams = qp[G.num_edges:-1]
    p = qp[-1]

    vs = vphi[0:G.num_edges]
    xis = vphi[G.num_edges:-1]
    phi = vphi[-1]


    ## Assemble variational formulation 

    # Initialize blocks in a and L to zero
    # (so fenics-mixed-dim does not throw an error)
    dx = Measure('dx', domain=mesh)
    a = Constant(0)*p*phi*dx
    for ix in range(0, len(G.bifurcation_ixs)):
        a += Constant(0)*lams[ix]*xis[ix]*dx
    L = f*phi*dx


    # Variational formulation
    
    # Assemble edge contributions to a and L
    for i, e in enumerate(G.edges):
        
        msh = G.edges[e]['submesh']
        vf = G.edges[e]['vf']
        
        dx_edge = Measure("dx", domain = msh)
        ds_edge = Measure('ds', domain=msh, subdomain_data=vf)

        # Add variational terms defined on edge
        a += (
                rho*Ainv*qs[i]*vs[i]*dx_edge
                + dt*mu*Ainv*G.dds(qs[i])*G.dds(vs[i])*dx_edge
                - dt*p*G.dds(vs[i])*dx_edge
                + G.dds(qs[i])*phi*dx_edge
                + dt*Res*Ainv*qs[i]*vs[i]*dx_edge
            )

        # Term from time derivative
        L += + rho*Ainv*qp_n.sub(i)*vs[i]*dx_edge


        # Add boundary condition for inflow/outflow boundary node
        for inlet_tag in inlets:
            L += Expression('x[0]', degree=2)*vs[i]*ds_edge(inlet_tag)
        for outlet_tag in outlets:
            L -= Expression('x[0]', degree=2)*vs[i]*ds_edge(outlet_tag)


    # Assemble vertex contribution to a, i.e. the bifurcation condition
    for i, b in enumerate(G.bifurcation_ixs):
        a += G.ip_jump_lm(qs, xis[i], b) + G.ip_jump_lm(vs, lams[i], b)
    
    qps = []
    for t in np.linspace(0, T, int(T/dt)):
        
        # Solve
        f.t = t

        qp_n1 = mixed_dim_fenics_solve(a, L, W, mesh)
        qps.append(qp_n1)

        # Update qp_ 
        for s in range(0, G.num_edges):
            assign(qp_n.sub(s), qp_n1.sub(s))
            
    return qps





if __name__ == '__main__':

    # Make simple graph
    from graph_examples import *
    G = honeycomb(1,1)
    G.make_mesh(1)

    fluid_params = {'rho':1, 'nu':1}
    f = Expression('sin(t)', degree=2, t=0)
    qps = network_stokes_model(G, fluid_params, f, inlets=[BOUN_OUT])


    filep =File('plots/p.pvd')
    fileq =File('plots/q.pvd')
    
    for i, qp in enumerate(qps):
        vars = qp.split()
        p = vars[-1]
        
        q = GlobalFlux(G, vars[0:-2])
        qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'DG', 2, G.geom_dim))
        p = vars[-1]
        
        qi.rename('q', '0.0')
        p.rename('p', '0.0')
        filep << (p,float(i))
        fileq << (qi,float(i))
        G.global_tangent.rename('tau', '0.0')
        File('plots/tangent.pvd')<<G.global_tangent

