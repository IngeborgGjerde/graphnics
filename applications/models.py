from tkinter import NE
import networkx as nx
from fenics import *
import sys
sys.path.append('../')
from graphnics import *


class HydraulicNetwork:
    '''
    Bilinear forms a and L for the hydraulic equations
            R*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous 
    normal stress
    
    Args:
        G (FenicsGraph): Network domain
        f (df.function): fluid source term
        ns (df.function): normal stress for neumann bcs
    '''
    
    def __init__(self, G, f, ns):
        self.G = G
        self.f = f
        self.ns = ns
        
    def a(self, qp, vphi):
        '''
        Args:
            qp (list of df.trialfunction)
            vphi (list of df.testfunction)
        '''
        
        G = self.G
        
        # split out the components
        qs, lams, p = qp[0:G.num_edges], qp[G.num_edges:-1], qp[-1]
        vs, xis, phi = vphi[0:G.num_edges], vphi[G.num_edges:-1], vphi[-1]
    
    
        dx = Measure('dx', domain=G.global_mesh)
        a = Constant(0)*p*phi*dx

        # Assemble edge contributions to a and L
        for i, e in enumerate(G.edges):
            
            Res = G.edges[e]['res']
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
            
            # Add variational terms defined on edge
            a += (
                    - p*G.dds(vs[i])*dx_edge
                    + G.dds(qs[i])*phi*dx_edge
                    + Res*qs[i]*vs[i]*dx_edge
                 )
            
        # Assemble vertex contribution to a, i.e. the bifurcation condition
        for i, b in enumerate(G.bifurcation_ixs):
            a += G.ip_jump_lm(qs, xis[i], b) + G.ip_jump_lm(vs, lams[i], b)
        
        return a 
   
    def L(self, vphi):
        '''
        Args:
            vphi (list of df.testfunction)
        '''
        
        G = self.G 
        
        # split out the components
        vs, xis, phi = vphi[0:G.num_edges], vphi[G.num_edges:-1], vphi[-1]
    
        dx = Measure('dx', domain=G.global_mesh)
        L = self.f*phi*dx

        # Assemble edge contributions to a and L
        for i, e in enumerate(G.edges):
            ds_edge = Measure('ds', domain=G.edges[e]['submesh'], subdomain_data=G.edges[e]['vf'])
            L += self.ns*vs[i]*ds_edge(BOUN_OUT) - self.ns*vs[i]*ds_edge(BOUN_IN)
       
        return L
    
    def deep_copy(self, f, ns):
        '''
        For time stepping we might need to evaluate a and L at different time steps
        
        To this end we make a deep copy where separate fs and ns can be used
        '''
        model_copy = HydraulicNetwork(self.G, f, ns)
        return model_copy



class NetworkStokes(HydraulicNetwork):
    '''
    Bilinear forms a and L for the network stokes equations
            mu*R/A q - mu/A d^2/ds^2 q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous 
    normal stress
    
    The linear form L is inherited from HydraulicNetwork.
    
    Args:
        G (FenicsGraph): Network domain
        fluid_params (dict): values for mu and rho
        f (df.function): fluid source term
        ns (df.function): normal stress for neumann bcs
 
    '''
    
    def __init__(self, G, fluid_params, f, ns):
        self.G = G
        self.fluid_params = fluid_params
        self.f = f
        self.ns = ns

    def a(self, qp, vphi):
        '''
        Args:
            qp (list of df.trialfunction)
            vphi (list of df.testfunction)
        '''
        
        G, mu = self.G, self.fluid_params["rho"]
        
        # split out the components
        qs, lams, p = qp[0:G.num_edges], qp[G.num_edges:-1], qp[-1]
        vs, xis, phi = vphi[0:G.num_edges], vphi[G.num_edges:-1], vphi[-1]
    
    
        dx = Measure('dx', domain=G.global_mesh)
        a = Constant(0)*p*phi*dx

        # Assemble edge contributions to a and L
        for i, e in enumerate(G.edges):
            
            Res, Ainv = [G.edges[e][key] for key in ['res', 'Ainv']]
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
            
            # Add variational terms defined on edge
            a += (
                    + mu*Ainv*G.dds(qs[i])*G.dds(vs[i])*dx_edge
                    - p*G.dds(vs[i])*dx_edge
                    + G.dds(qs[i])*phi*dx_edge
                    + mu*Ainv*Res*qs[i]*vs[i]*dx_edge
                )
            
        # Assemble vertex contribution to a, i.e. the bifurcation condition
        for i, b in enumerate(G.bifurcation_ixs):
            a += G.ip_jump_lm(qs, xis[i], b) + G.ip_jump_lm(vs, lams[i], b)
        
        return a
    
    def deep_copy(self, f, ns):
        '''
        For time stepping we might need to evaluate a and L at different time steps
        
        To this end we make a deep copy where separate fs and ns can be used
        '''
        model_copy = NetworkStokes(self.G, self.fluid_params, f, ns)
        return model_copy

   
   

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

    ## Assemble variational formulation 

    model = HydraulicNetwork(G, f, p_bc)
    
    a = model.a(qp, vphi)
    L = model.L(vphi)
    
    # Solve
    qp0 = mixed_dim_fenics_solve(a, L, W, mesh)
    return qp0

 
        


def time_dep_stokes(G, model, rho, t_steps, T, t_step_scheme = 'CN',
                   q0=None, p0=None):
    '''
    Solve models of the type 
        rho/A d/dt q + a((q,p,lam), (v,xi,phi)) = L(v,xi,phi;f,g,ns)
    on graph G
    
    Args:
        G (fg.FenicsGraph): problem domain
        model (class): class containing a and L
        fluid_params (dict): dict with values for rho
        t_steps (int): number of time steps
        T (float): end time
        t_step_scheme (str): time stepping scheme, either CN or IE
        
    The time stepping scheme can be set as "CN" (Crank-Nicholson) or IE (implicit Euler)
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
    qs, lams, p = qp[0:G.num_edges], qp[G.num_edges:-1], qp[-1]
    vs, xis, phi = vphi[0:G.num_edges], vphi[G.num_edges:-1], vphi[-1]

    qp_n = Function(W)
    qp_n_list = qp_n.split()
    q0s_, p0_  = qp_n_list[0:G.num_edges], qp_n_list[-1]
    for i, q0_ in enumerate(q0s_):
        q0_.assign(interpolate(q0, W.sub_space(i)))
    p0_.assign(interpolate(p0, W.sub_space(W.num_sub_spaces()-1)))
    
    dt = Constant(T/t_steps)
    dt_val = dt(0)
        
    qps = []
    
    if t_step_scheme is 'CN': 
        b2, b1 = Constant(0.5), Constant(0.5)
    else:
        b2, b1 = Constant(1), Constant(0)
        
        
    lhs_, rhs_ = 0, 0
    for i, e in enumerate(G.edges):
            
        msh = G.edges[e]['submesh']
        Ainv = G.edges[e]['Ainv']

        dx_edge = Measure("dx", domain = msh)
        
        lhs_ += rho*Ainv*qs[i]*vs[i]*dx_edge      
        rhs_ += rho*Ainv*qp_n.sub(i)*vs[i]*dx_edge


    # We make two copies of our model for time steps n and n+1
    f, ns = model.f, model.ns
    
    f.t, ns.t = 0, 0
    f_n, ns_n = [interpolate(func, FunctionSpace(mesh, 'CG', 3)) for func in [f, ns]]
    model_n = model.deep_copy(f_n, ns_n) #time step n
    
    f.t, ns.t = dt_val, dt_val
    f_n1, ns_n1 = [interpolate(func, FunctionSpace(mesh, 'CG', 3)) for func in [f, ns]]
    model_n1 = model.deep_copy(f_n1, ns_n1) #time step n+1
    
    # Get forms at each time
    a_form_n1, L_form_n1 = model_n1.a(qp, vphi), model_n1.L(vphi)   
    
    qp_n_list = [qp_n.sub(i) for i in range(0, W.num_sub_spaces())]    # TODO: clean this?
    a_form_n, L_form_n = model_n.a(qp_n_list, vphi), model_n.L(vphi)
        
   
    # Now we time step
    for t in np.linspace(dt_val, T, t_steps-1):
        
        lhs = lhs_ + b2*dt*a_form_n1
        rhs = rhs_ + b2*dt*L_form_n1 - b1*dt*a_form_n  + b1*dt*L_form_n
        
        qp_n1 = mixed_dim_fenics_solve(lhs, rhs, W, mesh)
        
        qps.append(qp_n1) 
            
        # Update qp_ 
        for s in range(0, W.num_sub_spaces()):
            assign(qp_n.sub(s), qp_n1.sub(s))
        
        # Update f and n at time steps n and n+1
           
        f_n, ns_n = [interpolate(func, FunctionSpace(mesh, 'CG', 3)) for func in [f, ns]]
        f.t, ns.t = t+dt_val, t+dt_val
        f_n1, ns_n1 = [interpolate(func, FunctionSpace(mesh, 'CG', 3)) for func in [f, ns]]
        
        model_n = model.deep_copy(f_n, ns_n) #time step n
        model_n1 = model.deep_copy(f_n1, ns_n1) #time step n+1
        
        # Get forms at each time
        a_form_n1, L_form_n1 = model_n1.a(qp, vphi), model_n1.L(vphi)   
        
        qp_n_list = [qp_n.sub(i) for i in range(0, W.num_sub_spaces())]    # TODO: clean this?
        a_form_n, L_form_n = model_n.a(qp_n_list, vphi), model_n.L(vphi)
        
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
        for e in G.in_edges(): G.in_edges()[e]['Res'] = 1
        for e in G.out_edges(): G.out_edges()[e]['Res'] = 1
        
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
    qp0 = Expression((sym.printing.ccode(q),sym.printing.ccode(p)), degree=2)
    
    print('****************************************************************')
    print('        Explicit computations         Via errornorm             ')
    print('h       ||q_e||_L2  ||p_e||_L2  |    ||q_e||_L2     ||p_e||_L2'  )
    print('****************************************************************')
    
    G = make_line_graph(4)
    G.make_mesh(8)

    G.neumann_inlets = [0]
    G.neumann_outlets = [1]
    
    for e in G.edges():
        G.edges[e]['res']=res
        G.edges[e]['Ainv']=Ainv

    model = NetworkStokes(G, fluid_params, f, ns)
    
    qps = time_dep_stokes(G, model, fluid_params, t_steps=30, T=1, qp0=qp0, f=f, g=g, ns=ns)

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

    #q_error = np.sum([errornorm(qh, qa) for qh, qa in zip(qhs, qas)]) 
    #pa2 = interpolate(p, FunctionSpace(G.global_mesh, 'CG', 3))
    #p_error = errornorm(ph, pa2)

    assert q_l2_error < 0.001, 'Network Stokes pressure solution not correct'
    assert p_l2_error < 0.001, 'Network Stokes flux solution not correct'



def convergence_test_stokes():
    ''''
    Test approximation of reduced stokes against analytic solution

    '''

    fluid_params = {'rho':1, 'nu':1}
    rho = fluid_params['rho']
    mu = rho*fluid_params['nu']

    res = 1
    Ainv = 1

    # We make the global q and global p smooth, so that the normal stress is continuous
    import sympy as sym
    x, t, x_ = sym.symbols('x[0] t x_')
    
    q = sym.cos(2*sym.pi*x) + sym.sin(2*sym.pi*t)
    f = q.diff(x)
    
    dsp = sym.diff(sym.diff(q, x), x) - q - sym.diff(q, t)
    p_ = sym.integrate(dsp, (x, 0, x_))
    p = p_.subs(x_, x)
    
    # Force source
    g = 0

    print('Analytic solutions')
    print('q', sym.printing.latex(q))
    print('p', sym.printing.latex(p))
    print('f', sym.printing.latex(f))
    
    # Normal stress
    ns = mu*Ainv*q.diff(x)-p 
    
    f, g, q, p, ns, q0, p0 = [Expression(sym.printing.ccode(func), degree=2, t=0) 
                                    for func in [f,g, q, p, ns, q, p]]
    
    
    print('*********************************')
    print('        Explicit computations    ')
    print('h       ||q_e||_L2  ||p_e||_L2   ')
    print('*********************************')
    for N in [0, 1, 2, 3, 4, 5]:

        G = make_line_graph(3)

        G.make_mesh(N)

        G.neumann_inlets = [0]
        G.neumann_outlets = [1]
        
        for e in G.edges():
            G.edges[e]['res']=res
            G.edges[e]['Ainv']=Ainv

        q.t = 0
        t_steps = 50
        model = NetworkStokes(G, fluid_params, f, ns)
        qps = time_dep_stokes(G, model, rho, t_steps=t_steps, T=1, q0=q0, p0=p0)

        vars = qps[-1].split(deepcopy=True)
        qhs = vars[0:G.num_edges]
        ph = vars[-1]

        pa2 = interpolate(p, FunctionSpace(G.global_mesh, 'CG', 3))
        p_error = errornorm(ph, pa2)
        q_error = 0
        for i, e in enumerate(G.edges()):
            qa = interpolate(q, FunctionSpace(G.edges[e]["submesh"], 'CG', 4))
            q_error += errornorm(qhs[i], qa)
            
        print(f'{G.global_mesh.hmin():1.3f}  &  {q_error:1.2e}  &   {p_error:1.2e}')
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(ph.function_space().tabulate_dof_coordinates()[:,0], ph.vector().get_local(), '.', label='h')
        plt.plot(pa2.function_space().tabulate_dof_coordinates()[:,0], pa2.vector().get_local(), '.', label='a')
        plt.legend()
        plt.savefig('sol-p.png')
        
        plt.figure()
        plt.plot(qhs[0].function_space().tabulate_dof_coordinates()[:,0], qhs[0].vector().get_local(), '.', label='h')
        plt.plot(qa.function_space().tabulate_dof_coordinates()[:,0], qa.vector().get_local(), '.', label='a')
        plt.legend()
        plt.savefig('sol-q.png')
        
        print(f'')



if __name__ == '__main__':
    #test_mass_conservation()
    #test_reduced_stokes()
    convergence_test_stokes()