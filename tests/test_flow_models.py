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
    Test mixed hydraulic network model against simple manufactured solution on Y bifurcation
    """

    G = make_Y_bifurcation()

    model = PrimalMixedHydraulicNetwork(G, p_bc = Expression('-x[1]', degree=2))
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
    nu = 2
    Ainv = 0.5
    Res = 10

    # We make the global q and global p smooth, so that the normal stress is continuous
    import sympy as sym

    x, x_ = sym.symbols("x[0] x_")

    q = sym.sin(2 * 3.14159 * x)
    f = q.diff(x)

    dsp = -Res*q + nu*Ainv*sym.diff(sym.diff(q, x), x)
    p_ = sym.integrate(dsp, (x, 0, x_))
    p = p_.subs(x_, x)

    g = nu*Ainv*sym.diff(f,x)

    # convert to fenics expressions
    f, q, p, g = [
        Expression(sym.printing.ccode(func), degree=2) for func in [f, q, p, g]
    ]

    # Solve on graph with single edge
    G = make_line_graph(2, dx=2)

    prop_dict = {
        key: {"Res": Constant(Res), "Ainv": Constant(Ainv)}
        for key in list(G.edges.keys())
    }
    nx.set_edge_attributes(G, prop_dict)

    
    G.make_mesh(10)
    model = MixedHydraulicNetwork(G, f=f, g=g, p_bc=p)
    sol = model.solve()
    print(sol)
    qh, ph = sol

    # Compute and print errors
    pa = interpolate(p, FunctionSpace(G.global_mesh, "CG", 2))
    p_error = errornorm(ph, pa)

    qa = interpolate(q, FunctionSpace(G.global_mesh, "CG", 3))
    q_error = errornorm(qh, qa)
        
    assert q_error < 1e-4, f"Network Stokes model not giving correct flux, q_error = {q_error}"
    assert p_error < 1e-1, f"Network Stokes model not giving correct pressure, p_error = {p_error}"


if __name__ == "__main__":
    test_network_stokes()
