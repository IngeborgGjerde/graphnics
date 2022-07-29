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



class DistFromSource(UserExpression):
    '''
    Evaluates the distance of a point on a graph to the source node
    '''

    def __init__(self, G, source_node, **kwargs):
        '''
        Args:
            G (nx.graph): Network graph
            source_node (int): 
        '''
        
        self.G=G
        self.source = source_node
        super().__init__(**kwargs)

        # If the edge lengths are not already computed, do that now
        if len(nx.get_edge_attributes(G, 'length')) is 0:
            G.compute_edge_lengths()

        # Get dict of nodes->dist from networkx
        dist = nx.shortest_path_length(G, 0, weight='length')
        
        # Store the distance value at each node as the dof values of a CG 1 function
        # Then for each point on the graph we get the linear interpolation of the nodal values
        # of the endpoint, which is exactly the distance at that point
        
        mesh = G.make_mesh(store_mesh = False, n=0) # we don't want to overwrite a pre-existing mesh
        
        V = FunctionSpace(mesh, 'CG', 1)
        dist_func = Function(V)
    

        # The vertices of the mesh are ordered like the nodes of the graph
        dofmap = list(dof_to_vertex_map(V)) # for going from mesh vertex to dof
        
        # Input nodal values in dist_func 
        for n in G.nodes():
            dof_ix = dofmap.index(n)
            dist_func.vector()[dof_ix] = dist[n]
        
        # Assign dist_func as a class variable and query it in eval
        self.dist_func = dist_func
        
        
    def eval(self, values, x):
        # Query the CG-1 dist_func 
        values[0] = self.dist_func(x)
        
        
        
        
        
def test_dist_from_source():
    
    # Test on simple line graph
    from graph_examples import make_line_graph
    G = make_line_graph(10)

    dist_from_source = DistFromSource(G, 0, degree=2)
    V = FunctionSpace(G.global_mesh, 'CG', 1)
    dist_from_source_i = interpolate(dist_from_source, V)

    coords = V.tabulate_dof_coordinates()
    lengths = np.linalg.norm(coords, axis=1)

    discrepancy = np.linalg.norm(np.asarray(dist_from_source_i.vector().get_local()) - np.asarray(lengths))
    assert near(discrepancy, 0)
    
    
    # Check on a double Y bifurcation
    from graph_examples import make_double_Y_bifurcation
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
        assert near(dist_from_source(G.nodes()[node]['pos']), dist), f'Distance not computed correctly for node {node}'



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
