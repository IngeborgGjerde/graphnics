import networkx as nx
from fenics import *
import sys
sys.path.append('../')
from graphnics import *
from applications.models import *

def convergence_test_stokes(t_step_scheme, t_steps=10, bifurcations=0):
    ''''
    Test approximation of reduced stokes against analytic solution

    Args:
        t_step_scheme (str): either 'CN' (Crank-Nicholson) or 'IE' (implicit Euler)
        t_steps (int): number of time steps
        bifurcations (int): number of bifurcations
    '''

    fluid_params = {'rho':1, 'nu':1}
    rho = fluid_params['rho']
    mu = rho*fluid_params['nu']
    fluid_params["mu"] = mu

    res = 1
    Ainv = 1

    # We make the global q and global p smooth, so that the normal stress is continuous
    import sympy as sym
    x, t, x_ = sym.symbols('x[0] t x_')
    
    q = sym.cos(2*sym.pi*x)# + sym.sin(2*sym.pi*t)
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

        G = make_line_graph(bifurcations+2)

        G.make_mesh(N)

        G.neumann_inlets = [0]
        G.neumann_outlets = [1]
        
        for e in G.edges():
            G.edges[e]['res']=res
            G.edges[e]['Ainv']=Ainv

        q.t = 0
        model = NetworkStokes(G, f=f, p_bc = ns)
        from time_stepping import time_stepping_stokes
        
        # TODO: Fix so that qp_n is the initial solution
        qps = time_stepping_stokes(model, rho, t_steps=t_steps, T=1, qp_n=None)

        vars = qps[-1]
        qhs = vars[0:G.num_edges]
        ph = vars[G.num_edges]

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
    import sys
    t_step_scheme = 'CN'#sys.argv[1]
    t_steps = 100#int(sys.argv[2])
    bifurcations = 1#int(sys.argv[3])
    
    convergence_test_stokes(t_step_scheme, t_steps, bifurcations)