import networkx as nx
from fenics import *
from fenics_graph import *
from utils import *
from graph_examples import *
    

@timeit
def hydraulic_network(G, f=Constant(0), p_bc=Constant(0)):
    '''
    Solve hydraulic network model 
        R q + d/ds p = 0
            d/ds q = f
    on graph G, with bifurcation condition q_in = q_out 

    Args:
        G (fg.FenicsGraph): problem domain
        f (df.function): source term
        p_bc (df.function): neumann bc for pressure
    '''
    
    
    mesh = G.global_mesh

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

    # split out the components
    qs = qp[0:G.num_edges]
    lams = qp[G.num_edges:-1]
    p = qp[-1]

    vs = vphi[0:G.num_edges]
    xis = vphi[G.num_edges:-1]
    phi = vphi[-1]

    
    ## Assemble variational formulation 

    # Initialize a and L to be zero
    dx = Measure('dx', domain=mesh)
    a = Constant(0)*p*phi*dx
    L = f*phi*dx

    # Assemble edge contributions to a and L
    for i, e in enumerate(G.edges):
        
        msh = G.edges[e]['submesh']
        vf = G.edges[e]['vf']
        #res = G.edges[e]['res']
        
        dx_edge = Measure("dx", domain = msh)
        ds_edge = Measure('ds', domain=msh, subdomain_data=vf)

        # Add variational terms defined on edge
        a += qs[i]*vs[i]*dx_edge        
        a -= p*G.dds(vs[i])*dx_edge
        a += phi*G.dds(qs[i])*dx_edge

        # Add boundary condition for inflow/outflow boundary node
        L += p_bc*vs[i]*ds_edge(BOUN_IN)
        L -= p_bc*vs[i]*ds_edge(BOUN_OUT)

    # Assemble vertex contribution to a, i.e. the bifurcation condition
    for i, b in enumerate(G.bifurcation_ixs):
        a += G.ip_jump_lm(qs, xis[i], b) + G.ip_jump_lm(vs, lams[i], b)
    
    
    # Solve
    qp0 = mixed_dim_fenics_solve(a, L, W, mesh)
    return qp0



def network_stokes(G, fluid_params, t_steps, T, q0=None, f=Constant(0), g=Constant(0), ns = Constant(0)):
    '''
    Solve reduced network Stokes model 
        rho/A d/dt q + mu*R/A q - mu/A d^2/ds^2 q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous 
    normal stress

    Args:
        G (fg.FenicsGraph): problem domain
        fluid_params (dict): dict with values for rho
        t_steps (int): number of time steps
        T (float): end time
        f (df.function): fluid source term
        g (df.function): force source term
        ns (df.function): normal stress for neumann bcs
    '''


    mesh = G.global_mesh
    rho = Constant(fluid_params['rho'])
    mu=Constant(fluid_params['nu'])*rho
    
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
    if q0:
        for i in range(0, G.num_edges):
            assign(qp_n.sub(i), interpolate(q0, W.sub_space(i)))

    # split out the components
    qs = qp[0:G.num_edges]
    lams = qp[G.num_edges:-1]
    p = qp[-1]

    vs = vphi[0:G.num_edges]
    xis = vphi[G.num_edges:-1]
    phi = vphi[-1]


    ## Assemble variational formulation 
    dt = Constant(T/t_steps)

    dx = Measure('dx', domain=mesh)
    a = Constant(0)*p*phi*dx
    L = f*phi*dx


    # Assemble edge contributions to a and L
    for i, e in enumerate(G.edges):
        
        msh = G.edges[e]['submesh']
        vf = G.edges[e]['vf']
        
        Res = G.edges[e]['res']
        Ainv = G.edges[e]['Ainv']

        dx_edge = Measure("dx", domain = msh)
        ds_edge = Measure('ds', domain=msh, subdomain_data=vf)

        # Add variational terms defined on edge
        a += (
                rho*Ainv*qs[i]*vs[i]*dx_edge
                + dt*mu*Ainv*G.dds(qs[i])*G.dds(vs[i])*dx_edge
                - dt*p*G.dds(vs[i])*dx_edge
                + G.dds(qs[i])*phi*dx_edge
                + dt*mu*Ainv*Res*qs[i]*vs[i]*dx_edge
            )

        # Term from time derivative
        L += + rho*Ainv*qp_n.sub(i)*vs[i]*dx_edge
        L += dt*dot(g,vs[i])*dx_edge


        # Add boundary condition for inflow/outflow boundary node
        L -= dt*ns*vs[i]*ds_edge(BOUN_IN)
        L += dt*ns*vs[i]*ds_edge(BOUN_OUT)


    # Assemble vertex contribution to a, i.e. the bifurcation condition
    for i, b in enumerate(G.bifurcation_ixs):
        a += G.ip_jump_lm(qs, xis[i], b) + G.ip_jump_lm(vs, lams[i], b)
    
    qps = []
    for t in np.linspace(0, T, t_steps):
       
        # Solve
        f.t = t
        g.t = t
        ns.t = t

        qp_n1 = mixed_dim_fenics_solve(a, L, W, mesh)
        
        qps.append(qp_n1)
        
            
        # Update qp_ 
        for s in range(0, G.num_edges):
            assign(qp_n.sub(s), qp_n1.sub(s))
            

    return qps







### ------------- Tests ------------ ###

def test_mass_conservation():
    '''
    Test mass conservation of hydraulic network model
    on double y bifurcation mesh and 1x1 honeycomb mesh
    '''
    
    Gs = []
    Gs.append(make_double_Y_bifurcation())
    Gs.append(honeycomb(1,1))


    for G in Gs:
        G.make_mesh(0)
        qp0 = hydraulic_network(G)

        vars = qp0.split()
        
        edges= list(nx.get_edge_attributes(G, 'submesh').keys())
        

        for b in G.bifurcation_ixs:
            flux_in, flux_out = 0, 0
            
            for e_in in G.in_edges(b):
                msh = G.edges[e_in]['submesh']
                vf = G.edges[e_in]['vf']
                b_ix = np.where(vf.array()==BIF_IN)[0]
                if b_ix:
                    b_coord = msh.coordinates()[b_ix[0],:] 
        
                    branch_ix = edges.index(e_in)
                    flux_in += vars[branch_ix](b_coord)
                
            for e_out in G.out_edges(b):
                msh = G.edges[e_out]['submesh']
                vf = G.edges[e_out]['vf']
                b_ix = np.where(vf.array()==BIF_OUT)[0]
                if b_ix:
                    b_coord = msh.coordinates()[b_ix[0],:] 
            
                    branch_ix = edges.index(e_out)
                    flux_out += vars[branch_ix](b_coord)
            
            assert near(flux_in, flux_out, 1e-3), f'Mass is not conserved at bifurcation {b}'



def test_reduced_stokes():
    ''''
    Test approximation of reduced stokes against analytic solution

    '''

    fluid_params = {'rho':1, 'nu':1}
    rho = fluid_params['rho']
    mu = rho*fluid_params['nu']

    res = 0.1
    Ainv = 2

    # We make the global q and global p smooth, so that the normal stress is continuous
    import sympy as sym
    x, t = sym.symbols('x[0] t')
    q = sym.sin(x) + sym.sin(t)
    p = sym.cos(x) + sym.cos(t)

    # Force source
    g = rho*Ainv*q.diff(t) + mu*Ainv*res*q
    g += - mu*Ainv*sym.diff(sym.diff(q,x),x) + p.diff(x)


    # Fluid source
    f = q.diff(x)

    # Normal stress
    ns = mu*Ainv*q.diff(x)-p 

    f, g, q, p, ns = [Expression(sym.printing.ccode(func), degree=2, t=0) for func in [f,g, q, p, ns]]
    
    
    print('****************************************************************')
    print('        Explicit computations         Via errornorm             ')
    print('h       ||q_e||_L2  ||p_e||_L2  |    ||q_e||_L2     ||p_e||_L2'  )
    print('****************************************************************')
    for N in [2, 3, 4, 5, 6]:

        G = make_line_graph(3)
        G.make_mesh(N)

        G.neumann_inlets = [0]
        G.neumann_outlets = [1]
        
        for e in G.edges():
            G.edges[e]['res']=res
            G.edges[e]['Ainv']=Ainv


        qps = network_stokes(G, fluid_params, t_steps=30, T=1, q0=q, f=f, g=g, ns=ns)

        p.t = 0.1
        q.t = 0.1

        vars = qps[0].split(deepcopy=True)
        qhs = vars[0:G.num_edges]
        ph = vars[-1]

        pa = interpolate(p, ph.function_space())
        qas = [interpolate(q, qh.function_space()) for qh in qhs]
        
        # The fenics-mixed-dim errornorm is a bit buggy
        # so we compute the errors manually and compare
        q_diffs = [(qas[i].vector().get_local()-qhs[i].vector().get_local())*G.global_mesh.hmin() for i in range(0,G.num_edges)]
        p_diff = (pa.vector().get_local()-ph.vector().get_local())*G.global_mesh.hmin()
        
        q_l2_error = np.sum([np.linalg.norm(q_diff) for q_diff in q_diffs])
        p_l2_error = np.linalg.norm(p_diff)

        q_error = np.sum([errornorm(qh, qa) for qh, qa in zip(qhs, qas)]) 
        pa2 = interpolate(p, FunctionSpace(G.global_mesh, 'CG', 3))
        p_error = errornorm(ph, pa2)
        
        print(f'{G.global_mesh.hmin():1.3f}   {q_l2_error:1.2e}     {p_l2_error:1.2e}     |    {q_error:1.2e}     {p_error:1.2e}')
        
        print(f'')


    assert q_l2_error < 0.01, 'Network Stokes pressure solution not correct'
    assert p_l2_error < 0.01, 'Network Stokes flux solution not correct'

if __name__ == '__main__':
    #test_mass_conservation()
    test_reduced_stokes()