
import networkx as nx
from fenics import *
from xii import *
import sys
sys.path.append('../../') 
from graphnics import *

time_stepping_schemes = {'IE':{'b1':Constant(0), 'b2':Constant(1)},
                         'CN':{'b1':Constant(0.5), 'b2':Constant(0.5)}}
    

class HydraulicNetwork_ii:
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
        
        # split out the components
        qs, lams, p = qp[0:self.G.num_edges], qp[self.G.num_edges:-1], qp[-1]
        vs, xis, phi = vphi[0:self.G.num_edges], vphi[self.G.num_edges:-1], vphi[-1]
        
        ## Init a as list of lists        
        a = [[0]*len(vphi)]*len(qp)
        
        # now fill in the non-zero blocks
        for i, e in enumerate(self.G.edges):
            
            resist = self.G.edges[e]['res']
            dx_edge = Measure("dx", domain = self.G.edges[e]['submesh'])
            
            Tp = Restriction(p, self.G.edges[e]['submesh'])
            Tphi = Restriction(phi, self.G.edges[e]['submesh'])
            
            a[i][i] += resist*qs[i]*vs[i]*dx_edge 
            
            A = ii_assemble(a)
            
            a[-1][i] += + qs[i]*Tphi*dx_edge
            print('>> Trying to assemble restricted element with global element')
            A = ii_assemble(a)
            
            a[i][-1] += - Tp*self.G.dds(vs[i])*dx_edge
            A = ii_assemble(a)
            
        
        assemble_mixed()
        
        # Add bifurcation condition
        edge_list = list(self.G.edges.keys())

        if False:
            for j, b in enumerate(self.G.bifurcation_ixs):
                lm_ix = self.G.num_edges + j
                
                for e in self.G.in_edges(j):
                    ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
                    edge_ix = edge_list.index(e)
                    a[lm_ix][edge_ix] += qs[edge_ix]*xis[j]*ds_edge(BIF_IN)
                    a[edge_ix][lm_ix] += vs[edge_ix]*lams[j]*ds_edge(BIF_IN)

                for e in self.G.out_edges(j):
                    ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
                    edge_ix = edge_list.index(e)
                    a[lm_ix][edge_ix] -= qs[edge_ix]*xis[j]*ds_edge(BIF_OUT)
                    a[edge_ix][lm_ix] -= vs[edge_ix]*lams[j]*ds_edge(BIF_OUT)
                
            
        return a 
   
    def L(self, vphi):
        '''
        Args:
            vphi (list of df.testfunction)
        '''
        
        # split out the components
        vs, xis, phi = vphi[0:self.G.num_edges], vphi[self.G.num_edges:-1], vphi[-1]
    
        dx = Measure('dx', domain=self.G.global_mesh)
        L = self.f*phi*dx

        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
            L += self.ns*vs[i]*ds_edge(BOUN_OUT) - self.ns*vs[i]*ds_edge(BOUN_IN)
       
        return L
    


def hydraulic_network_simulation(G, f=Constant(0), p_bc=Constant(0)):
    '''
    Set up spaces and solve hydraulic network model 
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
    W = P2s + LMs + [P1]
    
    # Trial and test functions
    qp = list(map(TrialFunction, W))
    vphi = list(map(TestFunction, W))
    
    ## Assemble variational formulation 

    model = HydraulicNetwork_ii(G, f, p_bc)
    
    a = model.a(qp, vphi)
    L = model.L(vphi)
    
    A = ii_assemble(a)
    b = ii_assemble(L)
    
    wh = ii_Function(W)
    solver = LUSolver(A, 'mumps')
    solver.solve(wh.vector(), b)

    return wh

 


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
        for e in G.in_edges(): G.in_edges()[e]['res'] = 1
        for e in G.out_edges(): G.out_edges()[e]['res'] = 1
        
        
        qp0 = hydraulic_network_simulation(G)

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



def convergence_test_stokes(t_step_scheme):
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
        qps = time_dep_stokes(G, model, rho, t_steps=t_steps, T=1, q0=q0, p0=p0, t_step_scheme=t_step_scheme)

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
    
    G = make_line_graph(2)
    G.make_mesh(2)
    for e in G.in_edges(): G.in_edges()[e]['res'] = 1
    for e in G.out_edges(): G.out_edges()[e]['res'] = 1
    
    qp0 = hydraulic_network_simulation(G)

    
    #test_mass_conservation()
    #test_reduced_stokes()
    #t_step_scheme = sys.argv[1]
    #convergence_test_stokes(t_step_scheme)