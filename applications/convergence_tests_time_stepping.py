from fenics import *
import sys
sys.path.append('../')
from graphnics import *
from applications.models import *
from time_stepping import time_stepping_stokes
        

def convergence_test_stokes(t_step_scheme, t_steps=10, T = 1, bifurcations=0):
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
    x, t, x_, pi = sym.symbols('x[0] t x_ pi')
    
    q = sym.sin(2*pi*x) + sym.cos(2*pi*t)
    f = q.diff(x)
    
    dsp = - q - sym.diff(q, t)# - sym.diff(sym.diff(q, x), x)
    p_ = sym.integrate(dsp, (x, 0, x_))
    p = p_.subs(x_, x)
    

    print('Analytic solutions')
    print('q =', sym.printing.latex(q))
    print('p =', sym.printing.latex(p))
    print('f =', sym.printing.latex(f))
    
    # Normal stress
    #ns = mu*Ainv*q.diff(x)-p 
    ns = -p 
    
    f, q, p, ns = [Expression(sym.printing.ccode(func), degree=2, t=0, pi=np.pi) for func in [f, q, p, ns]]
    
    
    print('*********************************')
    print('h       ||q_e||_L2  ||p_e||_L2   ')
    print('*********************************')
    for N in [1, 2, 3, 4, 5, 6]:

        G = make_line_graph(bifurcations+2)

        G.make_mesh(N)

        G.neumann_inlets = [0]
        G.neumann_outlets = [1]
        
        for e in G.edges():
            G.edges[e]['res']=res
            G.edges[e]['Ainv']=Ainv

        p.t, q.t, ns.t = 0, 0, 0
        model = HydraulicNetwork(G, f=f, p_bc = ns)

        qp_n = ii_Function(model.W)
        qp_a = [q]*G.num_edges + [p] + [ns]*G.num_bifurcations

        for i, func in enumerate(qp_a):
            qp_n[i].vector()[:] = interpolate(func, model.W[i]).vector().get_local()

        qps = time_stepping_stokes(model, rho, t_steps=t_steps, T=T, qp_n=qp_n, t_step_scheme=t_step_scheme)

        # Get final solution
        vars = qps[-1]
        qhs, ph = vars[0:G.num_edges], vars[G.num_edges]
        p.t, q.t = T, T

        # Compute and print errors
        pa = interpolate(p, FunctionSpace(G.global_mesh, 'CG', 2))
        p_error = errornorm(ph, pa)
        q_error = 0
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
    import sys
    t_step_scheme = sys.argv[1]
    t_steps = int(sys.argv[2])
    bifurcations = int(sys.argv[3])
    T = 1

    convergence_test_stokes(t_step_scheme, t_steps, T, bifurcations)