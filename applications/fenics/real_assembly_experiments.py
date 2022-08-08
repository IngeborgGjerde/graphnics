import networkx as nx
from fenics import *
from fenics_graph import *
from utils import *
from graph_examples import *
from utils import timeit
    
parameters["form_compiler"]["cpp_optimize"] = True

@timeit
def hydraulic_network_with_custom_assembly(G, f=Constant(0), p_bc=Constant(0)):
    '''
    Solve hydraulic network model 
        R q + d/ds p = 0
            d/ds q = f
    on graph G, with bifurcation condition q_in = q_out
    and custom assembly of the bifurcatio condition 

    Args:
        G (fg.FenicsGraph): problem domain
        f (df.function): source term
        p_bc (df.function): neumann bc for pressure
    '''
    
    mesh = G.global_mesh

    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    P2s = [FunctionSpace(msh, 'CG', 3) for msh in submeshes] 
    
    # Pressure space on global mesh
    P1 = FunctionSpace(mesh, 'CG', 2) # Pressure space (on whole mesh)
    
    ### Function spaces
    spaces = P2s + [P1]
    W = MixedFunctionSpace(*spaces) 


    # Trial and test functions
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)

    # split out the components
    qs = qp[0:G.num_edges]
    p = qp[-1]

    vs = vphi[0:G.num_edges]
    phi = vphi[-1]
    
    ## Assemble variational formulation 

    # Initialize blocks in a and L to zero
    # (so fenics-mixed-dim does not throw an error)
    dx = Measure('dx', domain=mesh)
    a = Constant(0)*p*phi*dx
    L = Constant(0)*phi*dx


    # Using methodology from firedrake we assemble the jumps as a vector
    # and input the jumps in the matrix later
    vecs = [[G.jump_vector(q, ix, j) for j in G.bifurcation_ixs] for ix, q in enumerate(qs)] 
    # now we can index by vecs[branch_ix][bif_ix]
    
    # Assemble edge contributions to a and L
    for i, e in enumerate(G.edges):
        
        msh = G.edges[e]['submesh']
        vf = G.edges[e]['vf']
        #res = G.edges[e]['res']
        
        dx_edge = Measure("dx", domain = msh)
        ds_edge = Measure('ds', domain=msh, subdomain_data=vf)

        # Add variational terms defined on edge
        a += qs[i]*vs[i]*dx_edge        
        a -= p*G.dds(vs[i])*dx_edge
        a += phi*G.dds(qs[i])*dx_edge

        # Add boundary condition for inflow/outflow boundary node
        L += p_bc*vs[i]*ds_edge(BOUN_IN)
        L -= p_bc*vs[i]*ds_edge(BOUN_OUT)
    
    # Solve
    qp0 = mixed_dim_fenics_solve_custom(a, L, W, mesh, vecs, G)
    return qp0


import argparse
import os
from applications.models.models import hydraulic_network_simulation

if __name__ == '__main__':
    '''
    Do time profiling for hydraulic network model implemented with fenics-mixed-dim
    Args:
        customassembly (bool): whether to assemble real spaces separately as vectors 
    customassembly should lead to speed up
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-customassembly', help='whether to do custom assembly of real spaces', default=True, type=str)
    args = parser.parse_args()

    if args.customassembly: 
        modelfunc = hydraulic_network_with_custom_assembly
    else: 
        modelfunc = hydraulic_network_simulation
    
    
    # Clear fenics cache
    print('Clearing cache')
    os.system('dijitso clean') 
    
    # Profile against a simple line graph with n nodes    
    n = 3
    G = make_line_graph(n)
    G.make_mesh(1) # we use just one cell per edge 
    mesh = G.global_mesh
    num_bifs = len(G.bifurcation_ixs)

    # Run with cache cleared and record times
    
    p_bc = Expression('x[0]', degree=1)
    
    qp0 = modelfunc(G, p_bc = p_bc)
    
    
    # Run again without clearing cache
    print('\n \n Running again with cache')
    
    qp0 = modelfunc(G, p_bc = p_bc)
        
    vars = qp0.split()
    p = vars[-1]
    
    q = GlobalFlux(G, vars[0:-1])
    qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'DG', 2, G.geom_dim))
    p = vars[-1]
    
    qi.rename('q', '0.0')
    p.rename('p', '0.0')
    File('plots/p.pvd')<<p
    File('plots/q.pvd')<<qi