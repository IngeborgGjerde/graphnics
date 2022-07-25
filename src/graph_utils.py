from fenics import *
import networkx as nx

def assign_radius_using_Murrays_law(G, start_node,  start_radius):
    '''
    Murray's law states that the thickness of branches in transport networks
    follow the relation
        r_p^3 = r_d1^3 + r_d2^3 + ... + r_dn^3
    where r_p is the radius of the parent and r_di is the radius of daughter vessel i.
    
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

            # find out how many edges the parent splits into
            connected_edges = G_.out_edges(v1)
            num_connected_edges = len(connected_edges) + DOLFIN_EPS 
            # TODO: why do we sometimes divide by zero here?
            
            # Now n*radius_dn**3 = radius_p**3 -> radius_dn = (1/n)**(1/3) radius_p
            radius_d = (1/num_connected_edges)**(1/3)*radius_p
            print(e, connected_edges, num_connected_edges, radius_p, radius_d)
            G_.edges()[e]['radius']=radius_d
            
    return G_



def test_Murrays_law_on_double_bifurcation():
    
    from graph_examples import make_double_Y_bifurcation
    G = make_double_Y_bifurcation()

    G = assign_radius_using_Murrays_law(G, start_node=0, start_radius=1)
        
    for v in G.nodes():
    es_in = list(G.in_edges(v))
    es_out = list(G.out_edges(v))
    
    # no need ot check terminal edges
    if len(es_out) is 0 or len(es_in) is 0: continue 
    
    r_p_cubed, r_d_cubed = 0, 0
    for e_in in es_in:
        r_p_cubed += G.edges()[e_in]['radius']**3
    for e_out in es_out:
        r_d_cubed += G.edges()[e_out]['radius']**3
    
    assert near(r_p_cubed, r_d_cubed, 1e-6), 'Murrays law not satisfied'
    