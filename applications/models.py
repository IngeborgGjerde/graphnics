
import networkx as nx
from fenics import *
from xii import *
import sys
sys.path.append('../') 
from graphnics import *


TH = {'flux_space': 'CG', 'flux_degree': 2, 'pressure_space': 'CG', 'pressure_degree': 1}
RT = {'flux_space': 'CG', 'flux_degree': 1, 'pressure_space': 'DG', 'pressure_degree': 0}

class HydraulicNetwork:
    '''
    Bilinear forms a and L for the hydraulic equations
            Res*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous 
    normal stress
    
    Args:
        G (FenicsGraph): Network domain
        Res (dict): dictionary with edge->resistance
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    '''
    
    def __init__(self, G, f=Constant(0), p_bc=Constant(0), space=RT):
        '''
        Set up function spaces and store model parameters f and ns
        '''
        
        # Graph on which the model lives
        self.G = G

        # Model parameters

        self.f = f
        self.p_bc = p_bc

        # Setup function spaces: 
        # - flux space on each segment, ordered by the edge list
        # - pressure space on the full mesh 
        # - real space on each bifurcation
        submeshes = list(nx.get_edge_attributes(G, 'submesh').values())

        P2s = [FunctionSpace(msh, space['flux_space'], space['flux_degree']) for msh in submeshes] 
        P1s = [FunctionSpace(G.global_mesh, space['pressure_space'], space['pressure_degree'])] 
        LMs = [FunctionSpace(G.global_mesh, 'R', 0) for b in G.bifurcation_ixs]

        ### Function spaces
        W = P2s + P1s + LMs
        self.W = W

        self.meshes = submeshes + [G.global_mesh]*(G.num_bifurcations+1) # associated meshes

        # Trial and test functions
        self.qp = list(map(TrialFunction, W))
        self.vphi = list(map(TestFunction, W))
     

    def diag_a_form_on_edges(self, a=None):
        '''
        Add edge contributions to the bilinear form
        '''

        if not a: a = self.init_a_form()
        
        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[:n_edges], vphi[:n_edges]
        
        # edge contributions to form
        for i, e in enumerate(G.edges):
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
    
            a[i][i] += self.G.edges()[e]["Res"]*qs[i]*vs[i]*dx_edge 
        
        return a


    def offdiag_a_form_on_edges(self, a=None):
        '''
        Add edge contributions to the bilinear form
        '''

        if not a: a = self.init_a_form()
        
        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[:n_edges], vphi[:n_edges]
        p, phi = qp[n_edges], vphi[n_edges]

        submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
        ps = [Restriction(p, msh) for msh in submeshes]
        phis = [Restriction(phi, msh) for msh in submeshes]

        # edge contributions to form
        for i, e in enumerate(G.edges):
            
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
            
            a[n_edges][i] += + G.dds_i(qs[i], i)*phis[i]*dx_edge
            a[i][n_edges] += - ps[i]*G.dds_i(vs[i], i)*dx_edge

        return a
        
        
    def a_form_on_bifs(self, a=None):
        '''
        Bifurcation point contributions to bilinear form a
        ''' 
        
        if not a: a = self.init_a_form()        
        
        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges
        
        qs, lams = qp[0:n_edges], qp[n_edges+1:]
        vs, xis = vphi[0:n_edges], vphi[n_edges+1:]

        edge_list = list(G.edges.keys())
        
        # bifurcation condition contributions to form
        for b_ix, b in enumerate(G.bifurcation_ixs):

            # make info dict of edges connected to this bifurcation
            conn_edges  = { **{e :  (1,  BIF_IN,  edge_list.index(e)) for e in G.in_edges(b) }, 
                            **{e :  (-1, BIF_OUT, edge_list.index(e)) for e in G.out_edges(b) }}
            
            for e in conn_edges:
                ds_edge = Measure('ds', domain=G.edges[e]['submesh'], subdomain_data=G.edges[e]['vf'])
                
                sign, tag, e_ix = conn_edges[e]

                a[e_ix][n_edges+1+b_ix] += sign*vs[e_ix]*lams[b_ix]*ds_edge(tag)
                a[n_edges+1+b_ix][e_ix] += sign*qs[e_ix]*xis[b_ix]*ds_edge(tag)
        
        return a 

    def init_a_form(self):
        '''
        Init a
        '''
        
        ## Init a as list of lists        
        a = [[ 0 for i in range(0, len(self.qp))  ] for j in range(0, len(self.qp))]

        # Init zero diagonal elements (for shape info)
        for i, msh in enumerate(self.meshes):
            a[i][i] += Constant(0)*self.qp[i]*self.vphi[i]*Measure('dx', domain=msh)

        return a


    def a_form(self):
        '''
        The bilinear form

        Args:
            Res (dict): dictionary with edge->resistance
        '''

        a = self.init_a_form()
        a = self.diag_a_form_on_edges(a)
        a = self.offdiag_a_form_on_edges(a)
        a = self.a_form_on_bifs(a)

        return a


    def init_L_form(self):
        '''
        Init L 
        '''
        
        L = [ 0 for i in range(0, len(self.vphi))  ]

        # Init zero diagonal elements (for shape info)
        for i, msh in enumerate(self.meshes):
            dx = Measure('dx', domain=msh)
            L[i] += Constant(0)*self.vphi[i]*dx

        return L

    def L_form(self):
        '''
        The right-hand side linear form
        '''
        
        L = self.init_L_form()

        vphi = self.vphi

        # split out the components
        n_edges = self.G.num_edges
        vs, phi, xis = vphi[0:n_edges], vphi[n_edges], vphi[n_edges+1:]

        submeshes = list(nx.get_edge_attributes(self.G, 'submesh').values())
        phis = [Restriction(phi, msh) for msh in submeshes]

        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
            dx_edge = Measure("dx", domain = self.G.edges[e]['submesh'])
            
            L[i] += self.p_bc*vs[i]*ds_edge(BOUN_OUT) - self.p_bc*vs[i]*ds_edge(BOUN_IN)
        
            L[n_edges] += self.f*phis[i]*dx_edge
        
        for i in range(0, len(self.G.bifurcation_ixs)):       
            L[n_edges+1+i] += Constant(0)*xis[i]*dx
        
        return L

    def B_form_eigval(self):
        '''
        The right-hand side linear form
        '''
        
        a = self.init_a_form()
        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[:n_edges], vphi[:n_edges]
        
        # edge contributions to form
        for i, e in enumerate(G.edges):
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
    
            a[i][i] += qs[i]*vs[i]*dx_edge 
       
        a[n_edges][n_edges] += qp[n_edges]*vphi[n_edges]*dx 
        
        return a





class NetworkStokes(HydraulicNetwork):
    '''
    Bilinear forms a and L for the hydraulic equations
            R*q + d^2/ds^2 q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous 
    normal stress
    
    Args:
        G (FenicsGraph): Network domain
        Res (dict): dictionary with edge->resistance
        nu (df.expr): viscosity
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    '''

    def __init__(self, G, mu=Constant(1), f=Constant(0), p_bc=Constant(0), space=TH):
        self.mu =mu
        super().__init__(G, f, p_bc, space)

    def diag_a_form_on_edges(self, a=None):
        '''
        The bilinear form
        '''

        if not a: a = self.init_a_form()

        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[0:n_edges], vphi[0:n_edges]

        # edge contributions to form
        for i, e in enumerate(G.edges):
            
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
            
            Res, Ainv = [self.G.edges()[e][key] for key in ['Res', 'Ainv']]
          
            a[i][i] += Res*qs[i]*vs[i]*dx_edge 
            a[i][i] += self.mu*Ainv*G.dds_i(qs[i],i)*G.dds_i(vs[i],i)*dx_edge 

        return a


### ------------- Tests ------------ ###

def test_mass_conservation():
    '''
    Test mass conservation of hydraulic network model at bifurcation points
    '''
    
    tests = {'Graph line' : make_line_graph(3, dim=3),
             'Y bifurcation': make_Y_bifurcation(),
             'YYY bifurcation': make_double_Y_bifurcation(),
             'honeycomb': honeycomb(4,4)
            }

    for test_name in tests:
        G = tests[test_name]
        
        G.make_mesh(5)

        model = HydraulicNetwork(G, p_bc=Expression('x[0]', degree=2))
        
        W = model.W
        a = model.a_form()
        L = model.L_form()
        
        A, b = map(ii_assemble, (a,L))
        A, b = map(ii_convert, (A,b))

        qp = ii_Function(W)
        solver = LUSolver(A, 'mumps')
        solver.solve(qp.vector(), b)

        edge_list = list(G.edges.keys())

        for b in G.bifurcation_ixs:
            total_flux = 0

            # make info dict of edges connected to this bifurcation
            conn_edges  = { **{e :  (1,  BIF_IN,  edge_list.index(e)) for e in G.in_edges(b) }, 
                            **{e :  (-1, BIF_OUT, edge_list.index(e)) for e in G.out_edges(b) }}
            
            for e in conn_edges:
                sign, tag, e_ix = conn_edges[e]
                total_flux += sign*qp[e_ix](G.nodes[b]['pos'])

            assert near(total_flux, 1e-3, 2e-3), f'Mass is not conserved at bifurcation {b} for {test_name}'



def spatial_convergence_test_stokes(bifurcations=1):
    ''''
    Test approximation of steady state reduced stokes against analytic solution
    '''

    # Model parameters
    rho = 10
    nu = 2
    mu = rho*nu
    Ainv = 0.5
    Res = 10

    # We make the global q and global p smooth, so that the normal stress is continuous
    import sympy as sym
    x, x_ = sym.symbols('x[0] x_')
    
    q = sym.sin(2*3.14159*x)
    f = q.diff(x)
    
    dsp = - Res*q + mu*Ainv*sym.diff(sym.diff(q, x), x)
    p_ = sym.integrate(dsp, (x, 0, x_))
    p = p_.subs(x_, x)
    
    ns = mu*Ainv*q.diff(x)-p # Normal stress
    
    print('Analytic solutions')
    print('q =', sym.printing.latex(q))
    print('p =', sym.printing.latex(p))
    print('f =', sym.printing.latex(f))
    
    f, q, p, ns = [Expression(sym.printing.ccode(func), degree=2) for func in [f, q, p, ns]]

    # Solve on increasingly fine meshes and record errors
    print('*********************************')
    print('h       ||q_e||_L2  ||p_e||_L2   ')
    print('*********************************')
    for N in [1, 2, 3, 4, 5]:
        
        G = make_line_graph(bifurcations+2)
        G.make_mesh(N)

        prop_dict = {key: { 'Res':Constant(Res),'Ainv':Constant(Ainv)} for key in list(G.edges.keys())}
        nx.set_edge_attributes(G, prop_dict)
    
        model = NetworkStokes(G, f=f, p_bc = ns, mu=Constant(mu))
    
        qp_n = ii_Function(model.W)
        qp_a = [q]*G.num_edges + [p] + [ns]*G.num_bifurcations

        for i, func in enumerate(qp_a):
            qp_n[i].vector()[:] = interpolate(func, model.W[i]).vector().get_local()

        A = ii_convert(ii_assemble(model.a_form()))
        b = ii_convert(ii_assemble(model.L_form()))
        A, b = [ii_convert(ii_assemble(term)) for term in [model.a_form(), model.L_form()]]
        
        sol = ii_Function(model.W)  
        solver = LUSolver(A, 'mumps')
        solver.solve(sol.vector(), b)

        qhs, ph = sol[:G.num_edges], sol[G.num_edges]

        # Compute and print errors
        pa = interpolate(p, FunctionSpace(G.global_mesh, 'CG', 2))
        p_error = errornorm(ph, pa)
        q_error = 0
        print(G.edges())
        for i, e in enumerate(G.edges()):
            qa = interpolate(q, FunctionSpace(G.edges[e]["submesh"], 'CG', 3))
            q_error += errornorm(qhs[i], qa)
            
        print(f'{G.global_mesh.hmin():1.3f}  &  {q_error:1.2e}  &   {p_error:1.2e}')
        
        import matplotlib.pyplot as plt
    
        # Plot solutions
        plt.figure()
        plt.plot(ph.function_space().tabulate_dof_coordinates()[:,0], ph.vector().get_local(), '*', label='h')
        plt.plot(pa.function_space().tabulate_dof_coordinates()[:,0], pa.vector().get_local(), '.', label='a')
        plt.legend()
        plt.savefig(f'sol-p.png')
        
        plt.figure()
        plt.plot(qhs[0].function_space().tabulate_dof_coordinates()[:,0], qhs[0].vector().get_local(), '*', label='h')
        plt.plot(qa.function_space().tabulate_dof_coordinates()[:,0], qa.vector().get_local(), '.', label='a')
        plt.legend()
        plt.savefig(f'sol-q.png')



if __name__ == '__main__':
    
    #test_mass_conservation()
    spatial_convergence_test_stokes()
