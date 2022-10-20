import networkx as nx
from fenics import *
import sys
sys.path.append('../')
set_log_level(40)
from graphnics import *

def test_mass_conservation():
    '''
    Test mass conservation of hydraulic network model at bifurcation points
    '''
    
    tests = {'Graph line' : make_line_graph(3, dim=3),
             'Y bifurcation': make_Y_bifurcation(),
             'YY bifurcation': make_double_Y_bifurcation(),
             'honeycomb': honeycomb(4,4)
            }

    for test_name in tests:
        G = tests[test_name]
        
        G.make_mesh(5)
        prop_dict = {key: { 'Res':Constant(1),'Ainv':Constant(1)} for key in list(G.edges.keys())}
        nx.set_edge_attributes(G, prop_dict)


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