import networkx as nx
from fenics import *
import sys

sys.path.append("../")

from graphnics import *


def test_fenics_graph():
    # Make simple y-bifurcation

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


def test_tangent_vector():
    # Make simple y-bifurcation
    G = make_Y_bifurcation()
    G.make_mesh(n=3)
    
    # Check that the first edge has tangent (0,1)
    first_tangent = G.edges()[(0,1)]['tangent']
    assert np.allclose(first_tangent, [0,1])
    
    # Check one of the tangents in a honeycomb network
    G = honeycomb(2,2)
    v1, v2 =  list(G.edges())[0]
    v1_pos = np.asarray(G.nodes()[v1]['pos'])
    v2_pos = np.asarray(G.nodes()[v2]['pos'])
    expected_tangent = np.asarray(v2_pos-v1_pos)
    expected_tangent *= 1/np.linalg.norm(expected_tangent)
    computed_tangent = G.edges()[(v1,v2)]['tangent']
    assert np.allclose(expected_tangent, computed_tangent)

if __name__ == "__main__":
    test_fenics_graph()
