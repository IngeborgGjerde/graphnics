'''
Copyright (C) 2022-2023 by Ingeborg Gjerde

This file is a part of the graphnics project (https://arxiv.org/abs/2212.02916)

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.
'''


from fenics import *
import networkx as nx
import numpy as np
import sys

sys.path.append("../")
from graphnics import *


def color_graph(G):
    """
    Args:
        G: networkx graph

    Colors the branches of a graph and stores the color (an int) as an edge variable.

    The graph colors can be displayed for 2d plots using
    >> plot_graph_color(G)

    """

    G_disconn = G.copy(as_view=False)
    G_disconn = nx.Graph(G_disconn)

    G_undir = nx.Graph(G)

    C = nx.adj_matrix(nx.Graph(G))
    num_vertex_conns = np.asarray(np.sum(C, axis=1)).flatten()
    bifurcation_points = np.where(num_vertex_conns > 2)[0].tolist()

    # To compute the branches we remove disconnect the edges meeting at a bifurcation point
    # For a bifurcation point b we create n new vertex copies b0 b1 b2 and connect each edge to
    # different vertex copies
    # We store the result in G_disconn

    for b in bifurcation_points:
        for i, e in enumerate(G_undir.edges(b)):

            # get the other node v2 connected to this edge
            vertex_list = list(e)
            vertex_list.remove(b)
            v2 = vertex_list[0]

            # we mark the new bifurcation vertex as 'b 0', 'b 1', ... etc
            new_bif_vertex = f"{b} {v2}"
            G_disconn.add_node(new_bif_vertex)
            G_disconn.nodes()[new_bif_vertex]["pos"] = G_undir.nodes()[b]["pos"]

            G_disconn.add_edge(new_bif_vertex, v2)  # a new disconnected edge

            # Remove the old edge
            try:
                # some edges consist of two bifurcation points, attempting to remove
                # it twice raises an error
                G_disconn.remove_edge(e[0], e[1])
            except:
                # sanity check: this should only be needed for edges consisting of two bpoints
                for v_e in e:
                    assert (
                        v_e in bifurcation_points
                    ), "Graph coloring algorithm may be malfunctioning"

    # From G_disconn we can get the disconnected subgraphs that each contain a branch
    subG = list(G_disconn.subgraph(c) for c in nx.connected_components(G_disconn))

    # Iterating over each subgraph we mark the edges with the branch number
    n_branches = 0

    for sG in subG:
        C = nx.adj_matrix(sG)
        num_vertex_conns = np.asarray(np.sum(C, axis=1)).flatten()

        # assert np.max(num_vertex_conns) < 3, 'subgraph contains a bifurcation when it should not'

        is_disconn_graph = (
            np.min(num_vertex_conns) > 0
        )  # some subgraphs are just points, no need to count those
        is_disconn_graph = is_disconn_graph and np.max(num_vertex_conns) < 3

        if is_disconn_graph:
            for e in sG.edges():
                v1 = str(e[0]).split(" ")[0]
                v2 = str(e[1]).split(" ")[0]

                # the graph G might be directed, so we look for (v1, v2) and (v2, v1)
                orig_e1 = (int(v1), int(v2))
                orig_e2 = (int(v2), int(v1))
                try:
                    G.edges()[orig_e1]["color"] = n_branches
                except:
                    G.edges()[orig_e2]["color"] = n_branches

            n_branches += 1

    # NOTE: One weakness of the above algorithm is that it does not color edges between two bifurcation points
    # so we have to color those manually at the end
    for e in G.edges():
        if "color" not in G.edges()[e]:
            G.edges()[e]["color"] = n_branches
            n_branches += 1


def plot_graph_color(G):
    """
    Plots the graph colorings stored as the edge dict entry 'color'
    """

    pos = nx.get_node_attributes(G, "pos")

    colors = nx.get_edge_attributes(G, "color")
    nx.draw_networkx_edge_labels(G, pos, colors)

    nx.draw_networkx(G, pos)
    colors = list(nx.get_edge_attributes(G, "color").values())


def assign_radius_using_Murrays_law(G, start_node, start_radius):
    """
    Assign radius information using Murray's law, which states that
    thickness of branches in transport networks follow the relation
        r_p^3 = r_d1^3 + r_d2^3 + ... + r_dn^3
    where r_p is the radius of the parent and r_di is the radius of daughter vessel i.
    The relative thickness of the daughter vessels is assigned proportional
    to the length of the subnetwork sprouting from it.

    Args:
        G: graph representing network
        start_node: input node to first edge in network
        start_radius: radius of that first edge

    Returns:
        A new graph G with a radius attribute on each edge

    The new graph has its edges sorted by a breadth-first-search starting at start_node
    """

    # number edges using breadth first search
    G_ = nx.bfs_tree(G, source=start_node)
    for i in range(0, len(G.nodes)):
        G_.nodes()[i]["pos"] = G.nodes()[i]["pos"]

    for e in list(G_.edges()):
        v1, v2 = e
        length = np.linalg.norm(
            np.asarray(G_.nodes()[v2]["pos"]) - np.asarray(G_.nodes()[v1]["pos"])
        )
        G_.edges()[e]["length"] = length

    assert (
        len(list(G.edges(start_node))) is 1
    ), "start node has to have a single edge sprouting from it"

    # iterate through the sorted edges, assigning radiuses as we go
    for i, e in enumerate(G_.edges()):

        # assign start radius to first edge
        if i == 0:
            G_.edges()[e]["radius"] = start_radius

        # compute radius for the rest
        else:
            v1, v2 = e  # edge goes from v1 to v2
            edge_in = list(G_.in_edges(v1))[0]
            radius_p = G_.edges()[edge_in]["radius"]

            # Compute the length fraction of the subgraph sprouting from this
            # daughters edge

            # find the subtree starting from v1 and then v2
            sub_graphs = {}
            sub_graph_lengths = {}
            for v in [v2, v1]:
                sub_graph = G_.subgraph(nx.shortest_path(G_, v))

                sub_graph_length = 0
                for d_e in sub_graph.edges():
                    sub_graph_length += sub_graph.edges()[d_e]["length"]

                sub_graphs[str(v)] = sub_graph
                sub_graph_lengths[str(v)] = sub_graph_length

            sub_graph_lengths[str(v2)] += G_.edges()[e]["length"]

            fraction = sub_graph_lengths[str(v2)] / sub_graph_lengths[str(v1)]

            if sub_graph_lengths[str(v2)] is 0:  # terminal edge
                fraction = 1 / len(sub_graphs[str(v1)].edges())

            # Now n*radius_dn**3 = radius_p**3 -> radius_dn = (1/n)**(1/3) radius_p
            radius_d = (fraction) ** (1 / 3) * radius_p
            G_.edges()[e]["radius"] = radius_d

    return G_


class DistFromSource(UserExpression):
    """
    Evaluates the distance of a point on a graph to the source node
    Computation assumes that the graph is non-cyclic
    """

    def __init__(self, G, source_node, **kwargs):
        """
        Args:
            G (nx.graph): Network graph
            source_node (int): Index of source node, i.e. the node from which the distance is measured
            
        Distances by traversing the graph in a breadth-first-search manner and adding up the edge lengths
        
        Results are stored as the dof values of a CG 1 function and can be used in variational formulations or queried directly
        >> dist = DistFromSource(G, source_node)
        >> print(dist(0.5, 0.5))
        
        """

        self.G = G
        self.source = source_node
        super().__init__(**kwargs)

        # If the edge lengths are not already computed, do that now
        if len(nx.get_edge_attributes(G, "length")) is 0:
            G.compute_edge_lengths()

        G_bfs = nx.bfs_tree(G, source_node)

        for n in G_bfs.nodes():
            G_bfs.nodes()[n]["pos"] = G.nodes()[n]["pos"]

        G_bfs = copy_from_nx_graph(G_bfs)
        G_bfs.compute_edge_lengths()

        # Get dict of nodes->dist from networkx
        dist = nx.shortest_path_length(G_bfs, source_node, weight="length")

        # Store the distance value at each node as the dof values of a CG 1 function
        # Then for each point on the graph we get the linear interpolation of the nodal values
        # of the endpoint, which is exactly the distance at that point

        mesh, mf = G_bfs.get_mesh(n=0)

        V = FunctionSpace(mesh, "CG", 1)
        dist_func = Function(V)

        # The vertices of the mesh are ordered like the nodes of the graph
        dofmap = list(dof_to_vertex_map(V))  # for going from mesh vertex to dof

        # Input nodal values in dist_func
        for n in G.nodes():
            dof_ix = dofmap.index(n)
            dist_func.vector()[dof_ix] = dist[n]

        # Assign dist_func as a class variable and query it in eval
        dist_func.set_allow_extrapolation(True)
        self.dist_func = dist_func
        

    def eval(self, values, x):
        # Query the CG-1 dist_func
        values[0] = self.dist_func(x)

    def value_shape(self):
        return ()
        