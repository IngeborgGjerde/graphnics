
import networkx as nx
from fenics import *
from xii import *
import sys
sys.path.append('../') 
from graphnics import *


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
    
    def __init__(self, G, Res=None, f=Constant(0), p_bc=Constant(0)):
        '''
        Set up function spaces and store model parameters f and ns
        '''
        
        # Graph on which the model lives
        self.G = G

        # Model parameters

        if Res is None:
            Res = { e:Constant(1) for e in self.G.edges() }
        self.Res = Res
        self.f = f
        self.p_bc = p_bc

        # Setup function spaces: 
        # - flux space on each segment, ordered by the edge list
        # - pressure space on the full mesh 
        # - real space on each bifurcation
        submeshes = list(nx.get_edge_attributes(G, 'submesh').values())

        P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] 
        P1s = [FunctionSpace(G.global_mesh, 'CG', 1)] 
        LMs = [FunctionSpace(G.global_mesh, 'R', 0) for b in G.bifurcation_ixs]
    
        ### Function spaces
        W = P2s + P1s + LMs
        self.W = W

        # Trial and test functions
        self.qp = list(map(TrialFunction, W))
        self.vphi = list(map(TestFunction, W))
            

    def add_form_edges(self, a):
        '''
        Add edge contributions to the bilinear form
        '''

        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[:n_edges], vphi[:n_edges]

        submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
        ps = [Restriction(qp[n_edges], msh) for msh in submeshes]
        phis = [Restriction(vphi[n_edges], msh) for msh in submeshes]

        # edge contributions to form
        for i, e in enumerate(G.edges):
            
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
            
            a[i][i] += self.Res[e]*qs[i]*vs[i]*dx_edge 
            a[n_edges][i] += + G.dds_i(qs[i], i)*phis[i]*dx_edge
            a[i][n_edges] += - ps[i]*G.dds_i(vs[i], i)*dx_edge

        return a
        
        
    def add_form_bifs(self, a):
        '''
        Bifurcation point contributions to bilinear form a
        ''' 

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

    def a_form(self):
        '''
        The bilinear form

        Args:
            Res (dict): dictionary with edge->resistance
        '''

        ## Init a as list of lists        
        a = [[ 0 for i in range(0, len(self.qp))  ] for j in range(0, len(self.qp))]

        a = self.add_form_edges(a)
        a = self.add_form_bifs(a)

        return a


    def L_form(self):
        '''
        The right-hand side linear form
        '''
        
        vphi = self.vphi

        # split out the components
        n_edges = self.G.num_edges
        vs, phi, xis = vphi[0:n_edges], vphi[n_edges], vphi[n_edges+1:]

        submeshes = list(nx.get_edge_attributes(self.G, 'submesh').values())
        phis = [Restriction(phi, msh) for msh in submeshes]

        
        L = [ 0 for i in range(0, len(vphi))  ]

        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
            dx_edge = Measure("dx", domain = self.G.edges[e]['submesh'])
            
            L[i] += self.p_bc*vs[i]*ds_edge(BOUN_OUT) - self.p_bc*vs[i]*ds_edge(BOUN_IN)
        
            L[n_edges+i] += self.f*phis[i]*dx_edge
        
        for i in range(0, len(self.G.bifurcation_ixs)):       
            L[n_edges+1+i] += Constant(0)*xis[i]*dx
        
        return L




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

    def __init__(self, G, Res=None, nu=Constant(1), f=Constant(0), p_bc=Constant(0)):
        self.nu = nu
        super().__init__(G, Res, f, p_bc)

    def add_form_edges(self, a):
        '''
        The bilinear form
        '''

        qp, vphi = self.qp, self.vphi
        G = self.G
        
        # split out the components
        n_edges = G.num_edges

        qs, vs = qp[0:n_edges], vphi[0:n_edges]

        submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
        ps = [Restriction(qp[n_edges], msh) for msh in submeshes]
        phis = [Restriction(vphi[n_edges], msh) for msh in submeshes]

        # edge contributions to form
        for i, e in enumerate(G.edges):
            
            dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
          
            a[i][i] += self.Res[e]*qs[i]*vs[i]*dx_edge 
            a[i][i] += self.nu*G.dds_i(qs[i],i)*G.dds_i(vs[i],i)*dx_edge 
            a[n_edges][i] += + G.dds_i(qs[i], i)*phis[i]*dx_edge
            a[i][n_edges] += - ps[i]*G.dds_i(vs[i], i)*dx_edge

        return a
       

    def a_form(self):
        '''
        The bilinear form

        Args:
            Res (dict): dictionary with edge->resistance
            nu (df.expr): 1d viscosity
        '''

        ## Init a as list of lists        
        a = [[ 0 for i in range(0, len(self.qp))  ] for j in range(0, len(self.qp))]

        a = self.add_form_edges(a)
        a = self.add_form_bifs(a)

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



def convergence_test_stokes(bifurcations=0):
    ''''
    Test approximation of steady state reduced stokes against analytic solution
'''

    fluid_params = {'rho':1, 'nu':1}
    rho = fluid_params['rho']
    mu = rho*fluid_params['nu']
    fluid_params["mu"] = mu

    # We make the global q and global p smooth, so that the normal stress is continuous
    import sympy as sym
    x, x_ = sym.symbols('x[0] x_')
    
    q = sym.sin(2*3.14*x)
    f = q.diff(x)
    
    dsp = - q # - sym.diff(sym.diff(q, x), x)
    p_ = sym.integrate(dsp, (x, 0, x_))
    p = p_.subs(x_, x)
    

    print('Analytic solutions')
    print('q =', sym.printing.latex(q))
    print('p =', sym.printing.latex(p))
    print('f =', sym.printing.latex(f))
    
    # Normal stress
    #ns = mu*Ainv*q.diff(x)-p 
    ns = -p 
    
    f, q, p, ns = [Expression(sym.printing.ccode(func), degree=2) for func in [f, q, p, ns]]

    print('*********************************')
    print('h       ||q_e||_L2  ||p_e||_L2   ')
    print('*********************************')
    for N in [1, 2, 3, 4]:
        
        G = make_line_graph(bifurcations+2)
        G.make_mesh(N)

        model = HydraulicNetwork(G, f=f, p_bc = ns)

        qp_n = ii_Function(model.W)
        qp_a = [q]*G.num_edges + [p] + [ns]*G.num_bifurcations

        for i, func in enumerate(qp_a):
            qp_n[i].vector()[:] = interpolate(func, model.W[i]).vector().get_local()

        a = model.a_form()
        L = model.L_form()
        A, b = [ii_convert(ii_assemble(term)) for term in [a, L]]
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
    convergence_test_stokes()
