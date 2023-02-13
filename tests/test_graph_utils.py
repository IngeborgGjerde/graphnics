import networkx as nx
from fenics import *
import sys

sys.path.append("../")
set_log_level(40)
from graphnics import *


def test_graph_color():

    # Check that a line graph gets all the same color
    G = make_line_graph(3)
    
    color_graph(G)
    assert G.edges()[(0, 1)]["color"] == G.edges()[(1, 2)]["color"]

    # check that a Y bifurcation gets three different colors
    G = make_Y_bifurcation()
    color_graph(G)
    colors = nx.get_edge_attributes(G, "color")
    assert len(set(colors.values())) == 3

    # check that 2x3 honeycomb gets 18 different colors

    G = honeycomb(2, 3)
    color_graph(G)

    colors = nx.get_edge_attributes(G, "color")
    num_colors = len(set(list(colors.values())))
    assert num_colors == 18

    # if something goes wrong you can plot the colors using
    # > plot_graph_color(G)


def test_dist_from_source():

    # Test on simple line graph
    G = make_line_graph(10)

    dist_from_source = DistFromSource(G, 0, degree=2)

    V = FunctionSpace(G.global_mesh, "CG", 1)
    dist_from_source_i = interpolate(dist_from_source, V)

    coords = V.tabulate_dof_coordinates()
    lengths = np.linalg.norm(coords, axis=1)

    discrepancy = np.linalg.norm(
        np.asarray(dist_from_source_i.vector().get_local()) - np.asarray(lengths)
    )
    print(dist_from_source_i.vector().get_local())
    print(np.asarray(lengths))
    assert near(discrepancy, 0), f"distance function discrepancy: {discrepancy}"

    # Check on a double Y bifurcation
    G = make_double_Y_bifurcation()

    dist_from_source = DistFromSource(G, 0, degree=2)

    source = 0

    for node in [3, 4, 5]:
        path = nx.shortest_path(G, source, node)[1:]

        v1 = source
        dist = 0
        for v2 in path:
            dist += G.edges()[(v1, v2)]["length"]
            v1 = v2
        assert near(
            dist_from_source(G.nodes()[node]["pos"]), dist
        ), f"Distance not computed correctly for node {node}"


def test_Murrays_law_on_double_bifurcation():

    G = make_double_Y_bifurcation()

    G = assign_radius_using_Murrays_law(G, start_node=0, start_radius=1)

    for v in G.nodes():
        es_in = list(G.in_edges(v))
        es_out = list(G.out_edges(v))

        # no need to check terminal edges
        if len(es_out) is 0 or len(es_in) is 0:
            continue

        r_p_cubed, r_d_cubed = 0, 0
        for e_in in es_in:
            r_p_cubed += G.edges()[e_in]["radius"] ** 3
        for e_out in es_out:
            r_d_cubed += G.edges()[e_out]["radius"] ** 3

        print(v, r_p_cubed, r_d_cubed)

        assert near(r_p_cubed, r_d_cubed, 1e-6), "Murrays law not satisfied"
    return G
