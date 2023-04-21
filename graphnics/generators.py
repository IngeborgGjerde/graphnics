'''
Copyright (C) 2022-2023 by Ingeborg Gjerde

This file is a part of the graphnics project (https://arxiv.org/abs/2212.02916)

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.
'''


import networkx as nx
from .fenics_graph import *


def line_graph(n, dim=2, dx=1):
    """
    Make a graph along the unit x-axis with n nodes

    Args:
        n (int): number of graph nodes
        dim (int): spatial dimension
        dx (float): distance between nodes
    """

    G = FenicsGraph()
    G.add_nodes_from(range(0, n))
    for i in range(0, n):
        G.nodes[i]["pos"] = [i * dx] + [0] * (dim - 1)  
        # coords is (dx, 0) in 2D and (dx, 0, 0) in 3D

    for i in range(0, n - 1):
        G.add_edge(i, i + 1)

    G.make_mesh()
    return G


def honeycomb(n, m):
    """
    Make honeycomb mesh with inlet and outlet
           
              /  (inlet)
           x x 
          /\/\
          x x x    (honeycomb)
          \/\/
           x x
(outlet) /
    
    Args:
        m (int): honeycomb rows
        n (int): honeycomb cols
    """

    # Make hexagonal mesh
    G = nx.hexagonal_lattice_graph(n, m)
    G = nx.convert_node_labels_to_integers(G)

    G.add_node(len(G.nodes))
    G.nodes[len(G.nodes) - 1]["pos"] = [0, -1]
    G.add_edge(len(G.nodes) - 1, 0)

    inlet_node = len(G.nodes) - 1

    # Add outlet
    # We want it positioned at the top right of the mesh
    pos = nx.get_node_attributes(G, "pos")
    all_coords = np.asarray(list(pos.values()))
    all_node_dist_from_origin = np.linalg.norm(all_coords, axis=1)
    furthest_node_ix = np.argmax(all_node_dist_from_origin, axis=0)
    coord_furthest_node = all_coords[furthest_node_ix, :]

    # Add new node a bit above the furthest one
    G.add_node(len(G.nodes))
    G.nodes[len(G.nodes) - 1]["pos"] = coord_furthest_node + np.asarray([0.7, 1])
    G.add_edge(len(G.nodes) - 1, furthest_node_ix)

    G = copy_from_nx_graph(G)

    # Usually the inlet edge is oriented outwards, we want it inwards
    if (0, inlet_node) in G.edges():
        G.remove_edge(0, inlet_node)
        G.add_edge(inlet_node, 0)

    G.make_mesh()
    return G


def Y_bifurcation(dim=2):
    """
    Generate a Y-bifurcation.
    
    Args:
        dim (int, optional): spatial dimension of node coords. Defaults to 2.

    Returns:
        FenicsGraph: tree network
    """
    G = FenicsGraph()

    G.add_nodes_from([0, 1, 2, 3])
    G.nodes[0]["pos"] = [0, 0] + [0]* (dim - 2)
    G.nodes[1]["pos"] = [0, 0.5]  + [0]* (dim - 2)
    G.nodes[2]["pos"] = [-0.5, 1]  + [0]* (dim - 2)
    G.nodes[3]["pos"] = [0.5, 1]  + [0]* (dim - 2)

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    G.make_mesh()
    return G


def YY_bifurcation(dim=2):
    """
    Generate a Y-bifurcation with two generations:
      Y Y
       Y
    
    Args:
        dim (int, optional): spatial dimension of node coords. Defaults to 2.

    Returns:
        FenicsGraph: tree network
    """
    
    G = FenicsGraph()

    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])  
    G.nodes[0]["pos"] = [0, 0] + [0]* (dim - 2)
    G.nodes[1]["pos"] = [0, 0.5] + [0]* (dim - 2)
    G.nodes[2]["pos"] = [-0.5, 1] + [0]* (dim - 2)
    G.nodes[3]["pos"] = [0.5, 1] + [0]* (dim - 2)

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    G.nodes[4]["pos"] = [-0.75, 1.5]+[0]* (dim - 2)
    G.nodes[5]["pos"] = [-0.25, 1.5]+[0]* (dim - 2)

    G.nodes[6]["pos"] = [0.25, 1.5]+[0]* (dim - 2)
    G.nodes[7]["pos"] = [0.75, 1.5]+[0]* (dim - 2)

    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(3, 6)
    G.add_edge(3, 7)

    G.make_mesh()
    return G
