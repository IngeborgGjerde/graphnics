from fenics import *
import sys
sys.path.append('../')
from graphnics import *
from applications.models import *
from time_stepping import time_stepping_stokes
import sympy as sym
import matplotlib.pyplot as plt

set_log_level(40) # disable warning messages in fenics
        

def spatial_convergence_test_stokes(Ns, mms, t_step_scheme, t_steps=10, T = 1, bifurcations=0):
    ''''
    Test approximation of reduced stokes against analytic solution

    Args:
        mms (dict): manufactured solution
        t_step_scheme (str): either 'CN' (Crank-Nicholson) or 'IE' (implicit Euler)
        t_steps (int): number of time steps
        bifurcations (int): number of bifurcations
    '''
    
    q, p, traction, f = [mms[var] for var in ['q', 'p', 'traction', 'f']]
    
    q_error_prev, p_error_prev, hprev = 1,1, 1
    
    print('*********************************')
    print('h       ||q_e||_H1  ||p_e||_L2   ')
    print('*********************************')

    for N in Ns:

        G = make_line_graph(bifurcations+2)

        prop_dict = {key: { 'Res':Constant(1),'Ainv':Constant(1)} for key in list(G.edges.keys())}
        nx.set_edge_attributes(G, prop_dict)
        
        G.make_mesh(N)

        p.t, q.t, traction.t = 0, 0, 0
        model = NetworkStokes(G, f=f, p_bc = traction)

        qp_n = ii_Function(model.W)
        qp_a = [q]*G.num_edges + [p] + [traction]*G.num_bifurcations

        for i, func in enumerate(qp_a):
            qp_n[i].vector()[:] = interpolate(func, model.W[i]).vector().get_local()

        qps = time_stepping_stokes(model, rho=1, t_steps=t_steps, T=T, qp_n=qp_n, t_step_scheme=t_step_scheme)

        # Get final solution
        vars = qps[-1]
        qhs, ph = vars[0:G.num_edges], vars[G.num_edges]
        p.t, q.t = T, T

        # Compute and print errors
        pa = interpolate(p, FunctionSpace(G.global_mesh, 'CG', 2))
        p_error = errornorm(ph, pa)
        
        # For flux error we sum H1 error on
        q_error = 0
        for i, e in enumerate(G.edges()):
            qa = interpolate(q, FunctionSpace(G.edges[e]["submesh"], 'CG', 3))
            q_error += errornorm(qhs[i], qa, 'H1')
        
        # Compute convergence rate
        hlog = np.log(G.global_mesh.hmin()) - np.log(hprev)
        q_conv_rate = (np.log(q_error)-np.log(q_error_prev) )/hlog   
        p_conv_rate = (np.log(p_error)-np.log(p_error_prev) )/hlog   
        
        q_error_prev, p_error_prev, hprev = q_error, p_error, G.global_mesh.hmin()

        # Now print        
        print(f'{G.global_mesh.hmin():1.3f} &  {q_error:1.2e}  ({q_conv_rate:1.1f}) &   {p_error:1.2e} ({p_conv_rate:1.1f}) \\\\')
        
        #plot_solution(ph, pa, qhs, qa)
    
            
            
def temporal_convergence_test_stokes(t_step_list, mms, t_step_scheme, T = 1, bifurcations=0):
    ''''
    Test approximation of reduced stokes against analytic solution

    Args:
        mms (dict): manufactured solution
        t_step_scheme (str): either 'CN' (Crank-Nicholson) or 'IE' (implicit Euler)
        t_steps (int): number of time steps
        bifurcations (int): number of bifurcations
    '''
    
    q, p, traction, f = [mms[var] for var in ['q', 'p', 'traction', 'f']]
    
    q_error_prev, p_error_prev, delta_t_prev = 1,1, 1
    
    G = make_line_graph(bifurcations+2)
    prop_dict = {key: { 'Res':Constant(1),'Ainv':Constant(1)} for key in list(G.edges.keys())}
    nx.set_edge_attributes(G, prop_dict)
    G.make_mesh(12)

    
    print('****************************************')
    print('delta_t       ||q_e||_H1  ||p_e||_L2   ')
    print('****************************************')

    for t_steps in t_step_list:

        p.t, q.t, traction.t = 0, 0, 0
        model = NetworkStokes(G, f=f, p_bc = traction)

        qp_n = ii_Function(model.W)
        qp_a = [q]*G.num_edges + [p] + [traction]*G.num_bifurcations

        for i, func in enumerate(qp_a):
            qp_n[i].vector()[:] = interpolate(func, model.W[i]).vector().get_local()

        qps = time_stepping_stokes(model, rho=1, t_steps=t_steps, T=T, qp_n=qp_n, t_step_scheme=t_step_scheme)

        # Get final solution
        vars = qps[-1]
        qhs, ph = vars[0:G.num_edges], vars[G.num_edges]
        p.t, q.t = T, T

        # Compute and print errors
        pa = interpolate(p, FunctionSpace(G.global_mesh, 'CG', 2))
        p_error = errornorm(ph, pa)
        
        # For flux error we sum H1 error on
        q_error = 0
        for i, e in enumerate(G.edges()):
            qa = interpolate(q, FunctionSpace(G.edges[e]["submesh"], 'CG', 3))
            q_error += errornorm(qhs[i], qa, 'H1')
        
        # Compute convergence rate
        delta_t = T/t_steps
        tlog = np.log(delta_t) - np.log(delta_t_prev)
        q_conv_rate = (np.log(q_error)-np.log(q_error_prev) )/tlog   
        p_conv_rate = (np.log(p_error)-np.log(p_error_prev) )/tlog   
        
        q_error_prev, p_error_prev, delta_t_prev = q_error, p_error, delta_t

        # Now print        
        print(f'{delta_t:1.3f} &  {q_error:1.2e}  ({q_conv_rate:1.1f}) &   {p_error:1.2e} ({p_conv_rate:1.1f}) \\\\')
        
        #plot_solution(ph, pa, qhs, qa)
    
        
def plot_solution(ph, pa, qhs, qa):
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
    
            


def time_dependent_mms(alpha):
    # We make the global q and global p smooth, so that the normal stress is continuous
    
    x, t, x_, pi = sym.symbols('x[0] t x_ pi')
    
    q = sym.sin(2*pi*x) + alpha*sym.cos(2*pi*t)
    f = q.diff(x)
    
    dsp = - q - sym.diff(q, t) + sym.diff(sym.diff(q, x), x)
    p_ = sym.integrate(dsp, (x, 0, x_))
    p = p_.subs(x_, x)
    
    nu, rho = 1, 1
    mu = rho*nu
    Ainv = 1
        
    print('Analytic solutions')
    print('q =', sym.printing.latex(q))
    print('p =', sym.printing.latex(p))
    print('f =', sym.printing.latex(f))
    
    # Normal stress
    traction = mu*Ainv*q.diff(x)-p 

    f, q, p, traction = [Expression(sym.printing.ccode(func), degree=2, t=0, pi=np.pi) for func in [f, q, p, traction]] 
    mms = {'q': q, 'p':p, 'traction':traction, 'f':f}
    
    return mms


if __name__ == '__main__':
    import sys
    t_steps = 500
    
    T = 1
 
    Ns = [2, 3, 4, 5, 6]
    
    time_disc_schemes = ['IE', 'CN']
    bifurcations = [1, 2]
    
    ###  Spatial discretization
    if False:
        print('\n\n Spatial discretization\n')
        
        ## Run stationary tests
        print('Running stationary tests')
        alpha=0
        mms = time_dependent_mms(alpha)

        for bps in bifurcations:                
            print(f'\nBifurcation points: {bps}')
            spatial_convergence_test_stokes(Ns, mms, t_step_scheme='IE', t_steps=2, T=T, bifurcations=bps)
        
    
    if False:    
        ## Run Implicit Euler and Crank-Nicholson tests
        
        for disc in time_disc_schemes:
            print(f'\n\nRunning {disc} tests')
            alpha=1
            mms = time_dependent_mms(alpha)
            
            for bps in bifurcations:                
                print(f'\nBifurcation points: {bps}')
                spatial_convergence_test_stokes(Ns, mms, t_step_scheme=disc, t_steps=500, T=T, bifurcations=bps)
            
    if True: 
        print('\n\n Time discretization\n')
        alpha=1
        mms = time_dependent_mms(alpha)
        
        for disc in time_disc_schemes:
            ###  Time discretization
            print(f'\n\nRunning {disc} tests')
        
            for bps in bifurcations:
                print(f'\nBifurcation points: {bps}')
                t_steps_list = [10, 20, 40, 80, 160]
                temporal_convergence_test_stokes(t_steps_list, mms, t_step_scheme=disc, T=T, bifurcations=bps)
                