import networkx as nx
from fenics import *
import sys

sys.path.append("../")
set_log_level(40)
from graphnics import *


def test_mass_conservation():
    """
    Test mass conservation of dual hydraulic network model at bifurcation points
    """

    tests = {
        "Graph line": make_line_graph(3, dim=3),
        "Y bifurcation": make_Y_bifurcation(),
        "YY bifurcation": make_double_Y_bifurcation(),
        "honeycomb": honeycomb(4, 4),
    }

    for test_name in tests:
        G = tests[test_name]

        G.make_mesh(5)
        prop_dict = {
            key: {"Res": Constant(1), "Ainv": Constant(1)}
            for key in list(G.edges.keys())
        }
        nx.set_edge_attributes(G, prop_dict)

        model = MixedHydraulicNetwork(G, p_bc=Expression("x[0]", degree=2))

        W = model.W
        a = model.a_form()
        L = model.L_form()

        A, b = map(ii_assemble, (a, L))
        A, b = map(ii_convert, (A, b))

        qp = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(qp.vector(), b)

        edge_list = list(G.edges.keys())

        for b in G.bifurcation_ixs:
            total_flux = 0

            # make info dict of edges connected to this bifurcation
            conn_edges = {
                **{e: (1, BIF_IN, edge_list.index(e)) for e in G.in_edges(b)},
                **{e: (-1, BIF_OUT, edge_list.index(e)) for e in G.out_edges(b)},
            }

            for e in conn_edges:
                sign, tag, e_ix = conn_edges[e]
                total_flux += sign * qp[e_ix](G.nodes[b]["pos"])

            assert near(
                total_flux, 1e-3, 2e-3
            ), f"Mass is not conserved at bifurcation {b} for {test_name}"


def test_hydraulic_network():
    """
    Test mixed hydraulic network model against manufactured solution on 
    """

    G = make_Y_bifurcation()

    model = HydraulicNetwork(G, p_bc = Expression('-x[1]', degree=2))
    q, p = model.solve()
    
    # Check mass conservation
    assert near(q(0,0), q(0.5, 1)+q(-0.5, 1), 1e-3)

    # Check that pressure bcs were applied
    assert near(p(0,0), 0)
    assert near(p(0.5, 1), -1)
    
    
    
def test_network_stokes():
    """'
    Test network stokes model against manufacture solution on single edge graph
    """

    # We check with parameters that are not one
    rho = 10
    nu = 2
    mu = rho * nu
    Ainv = 0.5
    Res = 10

    # We make the global q and global p smooth, so that the normal stress is continuous
    import sympy as sym

    x, x_ = sym.symbols("x[0] x_")

    q = sym.sin(2 * 3.14159 * x)
    f = q.diff(x)

    dsp = -Res * q + mu * Ainv * sym.diff(sym.diff(q, x), x)
    p_ = sym.integrate(dsp, (x, 0, x_))
    p = p_.subs(x_, x)

    ns = mu * Ainv * q.diff(x) - p  # Normal stress

    # convert to fenics expressions
    f, q, p, ns = [
        Expression(sym.printing.ccode(func), degree=2) for func in [f, q, p, ns]
    ]

    # Solve on graph with single edge
    G = make_line_graph(2)
    G.make_mesh(10)

    prop_dict = {
        key: {"Res": Constant(Res), "Ainv": Constant(Ainv)}
        for key in list(G.edges.keys())
    }
    nx.set_edge_attributes(G, prop_dict)

    model = NetworkStokes(G, f=f, p_bc=ns, mu=Constant(mu))

    A = ii_convert(ii_assemble(model.a_form()))
    L = model.L_form()
    b = ii_assemble(L)
    b = ii_convert(b)
    b = ii_convert(ii_assemble(model.L_form()))

    sol = ii_Function(model.W)
    solver = LUSolver(A, "mumps")
    solver.solve(sol.vector(), b)

    qh, ph = sol

    # Compute and print errors
    pa = interpolate(p, FunctionSpace(G.global_mesh, "CG", 2))
    p_error = errornorm(ph, pa)

    qa = interpolate(q, FunctionSpace(G.global_mesh, "CG", 3))
    q_error = errornorm(qh, qa)

    assert q_error < 1e-4, "Network Stokes model not giving correct flux"
    assert p_error < 1e-4, "Network Stokes model not giving correct pressure"


if __name__ == "__main__":
    test_network_stokes()
