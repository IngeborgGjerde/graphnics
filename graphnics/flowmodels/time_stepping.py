from fenics import *
from xii import *
import sys

from tqdm.notebook import tqdm

sys.path.append("../")
from graphnics import *

time_stepping_schemes = {"IE": {"b1": 0, "b2": 1}, "CN": {"b1": 0.5, "b2": 0.5}}

# TODO: 
# - Add simplified time stepping as an option to common script
# - Fix so that simplified time stepping also has t as a constant

def time_stepping_stokes(
    model, rho=Constant(1), t=Constant(0), t_steps=10, T=1, qp_n=None, t_step_scheme="IE"
):
    """
    Do time stepping for models of the type
        rho/A d/dt q + a(U, V) = L(V;f,g,ns)
    on graph G, where U=(q, ..., p) and V=(v, ..., phi)

    Args:
        model (class): class containing G, W, a and L
        rho (float): density
        t_steps (int): number of time steps
        T (float): end time
        qp_n (ii_function): initial solution
        t_step_scheme (str): 'IE' for Implicit Euler or 'CN' for Crank-Nicholson
    """

    if qp_n is None:
        qp_n = ii_Function(model.W)  # initialize as zero

    # split out the components
    qs = model.qp[: model.num_flux_spaces]
    vs = model.vphi[: model.num_flux_spaces]

    dt = T / t_steps

    # We discretize the time derivative term as rho/A d/dt q = rho/A (qn1 - qn)/Delta t
    cn, cn1 = time_stepping_schemes[t_step_scheme].values()
    
    print('Setting up problem...')
    Dn1 = model.init_a_form()
    Dn = model.init_L_form()

    for i, e in enumerate(model.G.edges):

        Ainv = model.G.edges()[e]["Ainv"]

        dx_edge = Measure("dx", domain=model.G.edges[e]["submesh"])

        Dn1[i][i] += rho * Ainv * qs[i] * vs[i] * dx_edge
        Dn[i] += rho * Ainv * qp_n[i] * vs[i] * dx_edge

    # Finally we time step
    qp_0 = ii_Function(model.W)
    for i, f0 in enumerate(qp_0):
        f0.vector()[:] = interpolate(qp_n[i], model.W[i]).vector().get_local()

    qps = [qp_0]

    a_diag = model.diag_a_form_on_edges()
    a_offdiag = model.offdiag_a_form_on_edges()
    a_bif = model.a_form_on_bifs()
    L = model.L_form()

    B = ii_convert(ii_assemble(a_offdiag))  # not time dependent
    C = ii_convert(ii_assemble(a_bif))  # not time dependent

    # Update f and n to next time step
    #model.f.t = t
    #model.p_bc.t = t
        
    An, Ln, DDn = [ii_convert(ii_assemble(term)) for term in [a_diag, L, Dn]]

    An1, Ln1, DDn1 = [ii_convert(ii_assemble(term)) for term in [a_diag, L, Dn1]]

    print('Time stepping')
    for t_val in tqdm(np.linspace(dt, T, t_steps - 1)):
        
        A = ii_convert(DDn1 + cn1 * dt * An1 + cn1 * dt * B + cn1 * dt * C)
        b = ii_convert(
            DDn + cn1 * dt * Ln1
         - dt*cn*An*qp_n.vector() - dt*cn*B*qp_n.vector() - cn*dt*C*qp_n.vector() + dt*cn*Ln )
        
        sol = ii_Function(model.W)
        solver = LUSolver(A, "mumps")
        solver.solve(sol.vector(), b)

        qps.append(sol)

        # Update qp_n
        [qp_n[i].assign(func) for i, func in enumerate(sol)]

        # Update f and n to next time step
        An = An1.copy()
        Ln = Ln1.copy()
        DDn = ii_convert(DDn1 * qp_n.vector())

        t.assign(t_val)

        #model.f.t = t
        #model.p_bc.t = t
    
        a_diag = model.diag_a_form_on_edges()
        a_offdiag = model.offdiag_a_form_on_edges()
        a_bif = model.a_form_on_bifs()
        L = model.L_form()
        
        An1 = ii_assemble(a_diag)
        Ln1 = ii_assemble(L)
        DDn1 = ii_assemble(Dn1)
              
        An1, Ln1, DDn1 = [ii_convert(term) for term in [An1, Ln1, DDn1]]

    return qps







def time_stepping_stokes_IE_simplified(
    model, rho=Constant(1), t=Constant(0), t_steps=10, T=1, qp_n=None, t_step_scheme="IE"
):
    """
    Do time stepping for models of the type
        rho/A d/dt q + a(U, V) = L(V;f,g,ns)
    on graph G, where U=(q, ..., p) and V=(v, ..., phi)

    Args:
        model (class): class containing G, W, a and L
        rho (float): density
        t_steps (int): number of time steps
        T (float): end time
        qp_n (ii_function): initial solution
        t_step_scheme (str): 'IE' for Implicit Euler or 'CN' for Crank-Nicholson
    """

    if qp_n is None:
        qp_n = ii_Function(model.W)  # initialize as zero
    

    # split out the components
    qs = model.qp[: model.G.num_edges]
    vs = model.vphi[: model.G.num_edges]

    dt = T / t_steps

    # We discretize the time derivative term as rho/A d/dt q = rho/A (qn1 - qn)/Delta t
    cn, cn1 = time_stepping_schemes[t_step_scheme].values()
    
    print('Setting up problem...')

    Dn1 = model.init_a_form()
    Dn = model.init_L_form()

    for i, e in enumerate(model.G.edges):

        Ainv = model.G.edges()[e]["Ainv"]

        dx_edge = Measure("dx", domain=model.G.edges[e]["submesh"])

        Dn1[i][i] += rho * Ainv * qs[i] * vs[i] * dx_edge
        Dn[i] += rho * Ainv * qp_n[i] * vs[i] * dx_edge

    # Finally we time step
    qp_0 = ii_Function(model.W)
    for i, f0 in enumerate(qp_0):
        f0.vector()[:] = interpolate(qp_n[i], model.W[i]).vector().get_local()

    qps = [qp_0]

    a_diag = model.diag_a_form_on_edges()
    a_offdiag = model.offdiag_a_form_on_edges()
    a_bif = model.a_form_on_bifs()
    L = model.L_form()

    B = ii_convert(ii_assemble(a_offdiag))  # not time dependent
    C = ii_convert(ii_assemble(a_bif))  # not time dependent

    # Update f and n to next time step
    model.f.t = t
    model.p_bc.t = t
        
    An, Ln, DDn = [ii_convert(ii_assemble(term)) for term in [a_diag, L, Dn]]

    Ln1 = ii_convert(ii_assemble(L))
    DDn1 = ii_convert(ii_assemble(Dn1))


    A = ii_convert(DDn1 + dt * An + dt * B + dt * C)
        
    print('Time stepping')
    for t in tqdm(np.linspace(dt, T, t_steps - 1)):
        
        b = ii_convert( DDn + dt * Ln1)
        
        sol = ii_Function(model.W)
        solver = LUSolver(A, "mumps")
        solver.solve(sol.vector(), b)

        qps.append(sol)

        # Update qp_n
        [qp_n[i].assign(func) for i, func in enumerate(sol)]

        # Update f and n to next time step
        DDn = ii_convert(DDn1 * qp_n.vector())

        model.f.t = t
        model.p_bc.t = t
    
        L = model.L_form()
        Ln1 = ii_convert(ii_assemble(L))

    return qps





