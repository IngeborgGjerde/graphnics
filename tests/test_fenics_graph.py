'''
Copyright (C) 2022-2023 by Ingeborg Gjerde

This file is a part of the graphnics project (https://arxiv.org/abs/2212.02916)

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.
'''

import networkx as nx
from fenics import *
from graphnics import *


def test_fenics_graph():
    # Make simple y-bifurcation

    G = Y_bifurcation()
    G.make_mesh()
    mesh = G.mesh

    # Check that all the mesh coordinates are also
    # vertex coordinates in the graph
    mesh_c = mesh.coordinates()
    for n in G.nodes:
        vertex_c = G.nodes[n]["pos"]
        vertex_ix = np.where((mesh_c == vertex_c).all(axis=1))[0]
        assert len(vertex_ix) == 1, "vertex coordinate is not a mesh coordinate"

def test_compute_vertex_degrees():
    G = Y_bifurcation()
    G.compute_vertex_degrees()

    degrees = nx.get_node_attributes(G, 'degree')
    assert near(degrees[0], 0.25)
    assert near(degrees[1], 0.9571, 0.01)


def test_tangent_vector():
    # Make simple y-bifurcation
    G = Y_bifurcation()
    G.make_mesh(n=3)
    G.make_submeshes()
    
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
