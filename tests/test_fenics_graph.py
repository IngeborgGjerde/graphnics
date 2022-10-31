import networkx as nx
from fenics import *
import sys

sys.path.append("../")

from graphnics import *


def test_fenics_graph():
    # Make simple y-bifurcation

    G = FenicsGraph()
    G = make_Y_bifurcation()
    G.make_mesh()
    mesh = G.global_mesh

    # Check that all the mesh coordinates are also
    # vertex coordinates in the graph
    mesh_c = mesh.coordinates()
    for n in G.nodes:
        vertex_c = G.nodes[n]["pos"]
        vertex_ix = np.where((mesh_c == vertex_c).all(axis=1))[0]
        assert len(vertex_ix) == 1, "vertex coordinate is not a mesh coordinate"


if __name__ == "__main__":
    test_fenics_graph()
