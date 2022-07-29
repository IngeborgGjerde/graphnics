from fenics import *
import networkx as nx
import numpy as np

def assign_radius_using_Murrays_law(G, start_node,  start_radius):
    '''
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
    '''
    
    
    # number edges using breadth first search
    G_ = nx.bfs_tree(G, source=start_node)
    for i in range(0, len(G.nodes)): 
        G_.nodes()[i]['pos'] = G.nodes()[i]['pos']
        
    for e in list(G_.edges()):
        v1, v2 = e
        length = np.linalg.norm( np.asarray(G_.nodes()[v2]['pos'])-np.asarray(G_.nodes()[v1]['pos']))
        G_.edges()[e]['length']=length
    
    assert len(list(G.edges(start_node))) is 1, 'start node has to have a single edge sprouting from it'
    
    # iterate through the sorted edges, assigning radiuses as we go
    for i, e in enumerate(G_.edges()):
        
        # assign start radius to first edge
        if i==0: 
            G_.edges()[e]['radius'] = start_radius
            
        # compute radius for the rest
        else:
            v1, v2 = e # edge goes from v1 to v2
            edge_in = list(G_.in_edges(v1))[0]
            radius_p = G_.edges()[edge_in]['radius']

            
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
            
            fraction = sub_graph_lengths[str(v2)]/sub_graph_lengths[str(v1)]   
            
            if sub_graph_lengths[str(v2)] is 0: # terminal edge
                fraction = 1/len(sub_graphs[str(v1)].edges())
                
            
            # Now n*radius_dn**3 = radius_p**3 -> radius_dn = (1/n)**(1/3) radius_p
            radius_d = (fraction)**(1/3)*radius_p
            G_.edges()[e]['radius']=radius_d
            
    return G_


def test_Murrays_law_on_double_bifurcation():
    
    from graph_examples import make_double_Y_bifurcation
    G = make_double_Y_bifurcation()

    G = assign_radius_using_Murrays_law(G, start_node=0, start_radius=1)
        
    for v in G.nodes():
        es_in = list(G.in_edges(v))
        es_out = list(G.out_edges(v))
        
        # no need to check terminal edges
        if len(es_out) is 0 or len(es_in) is 0: continue 
        
        r_p_cubed, r_d_cubed = 0, 0
        for e_in in es_in:
            r_p_cubed += G.edges()[e_in]['radius']**3
        for e_out in es_out:
            r_d_cubed += G.edges()[e_out]['radius']**3
        
        print(v, r_p_cubed, r_d_cubed)
        
        assert near(r_p_cubed, r_d_cubed, 1e-6), 'Murrays law not satisfied'
    return G